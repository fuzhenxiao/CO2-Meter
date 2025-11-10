import os
import time
import copy
import random
import torch
import logging
import argparse
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from dataset import CarbonDataset


class Metric(object):
    def __init__(self):
        self.all = self.init_pack()

    def init_pack(self):
        return {
            "cnt": 0,
            "apes": [],  # absolute percentage error
            "errbnd_cnt": np.array([0.0, 0.0, 0.0]),  # error bound count
            "errbnd_val": np.array(
                [0.1, 0.05, 0.01]
            ),  # error bound value: 0.1, 0.05, 0.01
        }

    def update_pack(self, ps, gs, pack):
        # print(ps)
        # print('lll')
        # print(gs)
        for i in range(len(ps)):
            ape = np.abs(ps[i] - gs[i]) / (gs[i]+1e-5)
            pack["errbnd_cnt"][ape <= pack["errbnd_val"]] += 1
            pack["apes"].append(ape)
        pack["cnt"] += len(ps)

    def measure_pack(self, pack):
        acc = np.mean(pack["apes"])
        err01 = (pack["errbnd_cnt"] / pack["cnt"])[0]
        err005 = (pack["errbnd_cnt"] / pack["cnt"])[1]
        err001 = (pack["errbnd_cnt"] / pack["cnt"])[2]
        return acc, err01,err005,err001, pack["cnt"]

    def update(self, ps, gs):
        self.update_pack(ps, gs, self.all)

    def get(self):
        return self.measure_pack(self.all)



class Trainer(object):

    def __init__(self):
        self.args = SimpleNamespace(
            lr=0.001,
            epochs=30,
            batch_size=1,
            momentum=0.9,
            weight_decay=1e-4,
            using_static_feature=False,
            print_freq=10,
            ckpt_save_freq=5,
            ckpt_save_dir='./',
            test_freq=5
            )

        self.logger = self.init_logger()
        self.logger.info("Loading args: \n{}".format(self.args))

        if not torch.cuda.is_available():
            self.logger.error("No GPU found!")
        else:
            self.logger.info("Using GPU")

        self.logger.info("Loading dataset:")
        self.train_loader, self.test_loader = self.init_dataset()

        self.logger.info("Loading model:")
        self.model, self.start_epoch, self.best_acc = self.init_model()
        self.best_err = 0
        print("Model:", self.model)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr
        )


    def init_logger(self):
        if not os.path.exists("log"):
            os.makedirs("log")
        logger = logging.getLogger("FEID")
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s-%(filename)s:%(lineno)d" "-%(levelname)s-%(message)s"
        )

        # log file stream
        handler = logging.FileHandler("default_log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        # log console stream
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)

        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def init_dataset(self):

        train_set = CarbonDataset(
            root="./processed_data/",
            input_file="xxx_energy_data.pt",
            global_file="xxx_global_feature.pt",
            length_file="xxx_length.txt",
            transform=None,
        )
        test_set = CarbonDataset(
            root="./processed_data/",
            input_file="xxx_energy_data.pt",
            global_file="xxx_global_feature.pt",
            length_file="xxx_length.txt",
            transform=None,
        )

        self.logger.info(
            "Train data = {}, Test data = {}".format(len(train_set), len(test_set))
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.args.batch_size, shuffle=True
        )

        test_loader = DataLoader(
            dataset=test_set, batch_size=self.args.batch_size, shuffle=False
        )

        return train_loader, test_loader


    def init_model(self):
        best_acc = 1e9
        start_epoch = 1
        from model import Net,FUGNN
        model=FUGNN()

        # if self.args.pretrain:
        #     self.logger.info("Loading pretrain: {}".format(self.args.pretrain))
        #     ckpt = torch.load(self.args.pretrain)
        #     model.load_state_dict(ckpt["state_dict"], strict=False)
        #     self.logger.info("Loaded pretrain: {}".format(self.args.pretrain))

        return model, start_epoch, best_acc

    def adjust_learning_rate(self, epoch):
        next_lr=0.001
        if epoch<30:
            for param_group in self.optimizer.param_groups:
                next_lr=self.args.lr
                param_group["lr"] = next_lr
        elif epoch<40:
            for param_group in self.optimizer.param_groups:
                next_lr=self.args.lr*0.1
                param_group["lr"] = next_lr
        else:
            for param_group in self.optimizer.param_groups:
                next_lr=self.args.lr*0.1
                param_group["lr"] = next_lr
        return next_lr


    def format_second(self, secs):
        return "Exa(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(
            int(secs / 3600), int((secs % 3600) / 60), int(secs % 60)
        )

    def save_checkpoint(self, epoch, best=False):
        epoch_str = "best" if best else "e{}".format(epoch)
        model_path = "{}ckpt_{}.pth".format(self.args.ckpt_save_dir, epoch_str)

        if not os.path.exists(self.args.ckpt_save_dir):
            os.makedirs(self.args.ckpt_save_dir)

        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "best_acc": self.best_acc,
            },
            model_path,
        )

        self.logger.info("Checkpoint saved to {}".format(self.args.ckpt_save_dir))
        return

    def train_epoch(self, epoch):
        self.model.train()
        t0 = time.time()
        metric = Metric()

        lr = self.adjust_learning_rate(epoch)
        num_iter = len(self.train_loader)

        for iteration, batch in enumerate(self.train_loader):
            torch.cuda.empty_cache()

            data, static_feature = batch
            data.y = data.y.view(-1, 1)
            data = data.to(self.device)
            static_feature = static_feature.to(self.device)

            self.optimizer.zero_grad()
            pred_cost = self.model(data, static_feature)

            loss = F.mse_loss(pred_cost / data.y, data.y / data.y)
            loss.backward()
            self.optimizer.step()

            ps = pred_cost.data.cpu().numpy()[:, 0].tolist()
            gs = data.y.data.cpu().numpy()[:, 0].tolist()

            metric.update(ps, gs)
            acc, err01,err005,err001, cnt = metric.get()

            if iteration % self.args.print_freq == 0:
                t1 = time.time()
                speed = (t1 - t0) / (iteration + 1)
                exp_time = self.format_second(
                    speed * (num_iter * (self.args.epochs - epoch + 1) - iteration)
                )

                self.logger.info(
                    "Epoch[{}/{}]({}/{}) Lr:{:.8f} Loss:{:.5f} MAPE:{:.5f} "
                    "ErrBnd(0.1):{:.5f},(0.05):{:.5f},(0.01):{:.5f}".format(
                        epoch,
                        self.args.epochs,
                        iteration,
                        num_iter,
                        lr,
                        loss.data,
                        acc,
                        err01,
                        err005,
                        err001
                    )
                )
        return acc, err01,err005,err001        

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.args.only_test = False
            self.train_epoch(epoch)

            if epoch > 0 and epoch % self.args.ckpt_save_freq == 0:
                self.save_checkpoint(epoch)

            if epoch > 0 and epoch % self.args.test_freq == 0:
                acc, err01,err005,err001 = self.test()
                if acc < self.best_acc:
                    self.best_acc = acc
                    self.best_err = err01
                    self.save_checkpoint(epoch, best=True)
        self.logger.info(
            "Train over, best acc = {:.5f}, err01 = {}".format(
                self.best_acc, self.best_err
            )
        )
        return




    def test(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()
        t0 = time.time()
        num_iter = len(self.test_loader)
        if num_iter <= 0:
            return 0, 0
        metric = Metric()

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                torch.cuda.empty_cache()

                data, static_feature = batch
                data.y = data.y.view(-1, 1)
                data = data.to(self.device)
                static_feature = static_feature.to(self.device)
                pred_cost = self.model(data, static_feature)

                ps = pred_cost.data.cpu().numpy()[:, 0].tolist()

                gs = data.y.data.cpu().numpy()[:, 0].tolist()
                metric.update(ps, gs)
                acc, err01,err005,err001, cnt = metric.get()

                if iteration > 0 and iteration % 50 == 0:
                    self.logger.info(
                        "[{}/{}] MAPE: {:.5f} ErrBnd(0.1): {:.5f},(0.05):{:.5f},(0.01):{:.5f}".format(
                            iteration, num_iter, acc, err01,err005,err001
                        )
                    )

            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000
            acc, err01,err005,err001, cnt = metric.get()

            self.logger.info(
                " ------------------------------------------------------------------"
            )
            self.logger.info(" * Speed: {:.5f} ms/iter".format(speed))
            self.logger.info(" * MAPE: {:.5f}".format(acc))
            self.logger.info(" * ErrorBound (0.1): {},(0.05):{:.5f},(0.01):{:.5f}".format(err01,err005,err001))
            self.logger.info(
                " ------------------------------------------------------------------"
            )
            return acc, err01,err005,err001 



class Tester(object):

    def __init__(self,model_path,test_data_path):


        self.model_path=model_path
        self.test_data_path=test_data_path
        self.test_loader = self.init_dataset()

        self.model= self.init_model()
        print("Model:", self.model)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

    def init_dataset(self):


        test_set = CarbonDataset(
            root="./processed_data/",
            input_file="xxx_energy_data.pt",
            global_file="xxx_global_feature.pt",
            length_file="xxx_length.txt",
            transform=None,
        )

        test_loader = DataLoader(
            dataset=test_set, batch_size=1, shuffle=False
        )

        return test_loader


    def init_model(self):
        best_acc = 1e9

        from model import Net,FUGNN
        model=FUGNN()

        ckpt = torch.load(self.model_path)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

    def test(self):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        self.model.eval()
        t0 = time.time()
        num_iter = len(self.test_loader)
        if num_iter <= 0:
            return 0, 0
        metric = Metric()

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                torch.cuda.empty_cache()

                data, static_feature = batch
                data.y = data.y.view(-1, 1)
                data = data.to(self.device)
                static_feature = static_feature.to(self.device)
                pred_cost = self.model(data, static_feature)

                # pred_cost = torch.exp(pred_cost)
                ps = pred_cost.data.cpu().numpy()[:, 0].tolist()

                gs = data.y.data.cpu().numpy()[:, 0].tolist()
                metric.update(ps, gs)
                acc, err01,err005,err001, cnt = metric.get()

                if iteration > 0 and iteration % 50 == 0:
                    print(
                        "[{}/{}] MAPE: {:.5f} ErrBnd(0.1): {:.5f},(0.05):{:.5f},(0.01):{:.5f}".format(
                            iteration, num_iter, acc, err01,err005,err001
                        )
                    )

            t1 = time.time()
            speed = (t1 - t0) / num_iter * 1000
            acc, err01,err005,err001,cnt = metric.get()

            print(
                " ------------------------------------------------------------------"
            )
            print(" * MAPE: {:.5f}".format(acc))
            print(" * ErrorBound (0.1): {},(0.05):{:.5f},(0.01):{:.5f}".format(err01,err005,err001))
            print(
                " ------------------------------------------------------------------"
            )


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    #tester = Tester(model_path='./pretrained_models/code/trace/ckpt_best.pth',test_data_path=None)
    #tester.test()