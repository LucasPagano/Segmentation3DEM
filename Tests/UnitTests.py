import imageio
import numpy as np
import os
import sys
import time
import torch
import wandb

from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.getcwd()))

from Tools import Dotdict
import ImageDataGenerator
import ImagesIO
import ImageTools
import Normalizers
import Processing

import FiReTiTiPyTorchLib.FiReTiTiPyTorchLib_Losses as Losses

import PyTorch_Models.ResUnetA.ResUnetA as ResUnetA

Dashes = " -------------------------------------------- "


def CreateDirs(Name: str, GT: bool = False):
    os.makedirs(Name + "/In/", exist_ok=True)
    if GT:
        os.makedirs(Name + "/GT/", exist_ok=True)


def TestKerasClassification1():
    print(Dashes + "Starting TestKerasClassification1" + Dashes)
    results = "Results Keras Classification 1"
    CreateDirs(results)

    dim = 256
    minus = 0
    BatchSize = 6
    CropPerImage = 2

    inputs = "/Users/firetiti/NN/DataSets/Test Classification/"

    generator = ImageDataGenerator.Generator(ChannelFirst=True)
    generator.setCropPerImage(CropPerImage)
    generator.setBrighterDarker(31, "Uniform_PerChannel")
    generator.LoadInputs(inputs, OnTheFly=True, Classification=True)
    generator.setInputsDimensions(dim, dim)
    generator.setOutputsDimensions(dim - minus, dim - minus)
    gen = generator.Keras(BatchSize, InputsNormalizer=None, OutputsNormalizers=None)

    nbEpoch = 2
    for epoch in range(nbEpoch):
        index = 0
        for batch in range(gen.getNbBatchPerEpoch()):
            X, Y = gen.next()

            for i in range(len(X)):
                prefix = str(epoch) + " - Batch " + str(batch) + " - Input " + str(i)
                Processing.ForceRange(X[i])
                ImagesIO.Write(X[i], True, results + "/In/Epoch " + prefix + " - " + str(index) + ".png")
                index += 1
    print("TestKerasClassification1 Done!\n\n\n\n\n")


def TestKerasClassification2():
    print(Dashes + "Starting TestKerasClassification2" + Dashes)
    results = "Results Keras Classification 2"
    CreateDirs(results)

    dim = 248
    minus = 10
    BatchSize = 9

    inputs = "/Users/firetiti/NN/DataSets/Test Classification/"

    InputsNormalizer = Normalizers.Normalize()

    generator = ImageDataGenerator.Generator(ChannelFirst=True)
    generator.setShuffle(True)
    generator.setFlip(True)
    generator.setRotate90x(True)
    generator.setMaxShiftRange(1000)
    generator.setNoise(31, "Gaussian", 51)
    generator.LoadInputs(inputs, OnTheFly=False, Classification=True)
    generator.setInputsDimensions(dim, dim)
    gen = generator.Keras(BatchSize, InputsNormalizer=InputsNormalizer, OutputsNormalizers=None)

    nbEpoch = 2
    for epoch in range(nbEpoch):
        index = 0
        for batch in range(gen.getNbBatchPerEpoch()):
            X, Y = gen.next()

            InputsNormalizer.Denormalize(X)

            for i in range(len(X)):
                prefix = str(epoch) + " - Batch " + str(batch) + " - Input " + str(i)
                ImagesIO.Write(X[i], True, results + "/In/Epoch " + prefix + " - Input.png")
                index += 1
    print("TestKerasClassification2 Done!\n\n\n\n\n")


def TestKerasSegmentation1():
    print(Dashes + "Starting TestKerasSegmentation1" + Dashes)
    results = "Results Keras Segmentation 1"
    CreateDirs(results, GT=True)

    dim = 256
    minus = 10
    BatchSize = 10
    CropPerImage = 2

    inputs = "/Users/firetiti/NN/DataSets/Test Segmentation Inputs/"
    outputs = "/Users/firetiti/NN/DataSets/Test Segmentation Outputs/"

    InputsNormalizers = [Normalizers.Normalize(), Normalizers.Normalize()]
    OutputsNormalizers = [Normalizers.Basic(), Normalizers.Basic()]

    generator = ImageDataGenerator.Generator(ChannelFirst=True)
    generator.setCropPerImage(CropPerImage)
    generator.setMaxShiftRange(1000)
    generator.LoadInputs(inputs, OnTheFly=True, Classification=False)
    generator.LoadOutputs(outputs)
    generator.setInputsDimensions(dim, dim)
    generator.setOutputsDimensions(dim - minus, dim - minus)
    gen = generator.Keras(BatchSize, InputsNormalizers=InputsNormalizers, OutputsNormalizers=OutputsNormalizers)

    nbEpoch = 2
    for epoch in range(nbEpoch):
        for batch in range(gen.getNbBatchPerEpoch()):
            X, Y = gen.next()

            InputsNormalizers[0].Denormalize(X)
            OutputsNormalizers[0].Denormalize(Y)

            index = 0
            for inputs, outputs in zip(X, Y):
                prefix = str(epoch) + " - Batch " + str(batch) + " - Dir " + str(index) + " - "
                for i in range(inputs.shape[0]):
                    ImagesIO.Write(inputs[i], True, results + "/In/Epoch " + prefix + str(i) + " - Input.png")
                    ImagesIO.Write(outputs[i][0], True, results + "/GT/Epoch " + prefix + str(i) + " - Output.png")
                index += 1

    print("TestKerasSegmentation1 Done!\n\n\n\n\n")


def TestKerasSegmentation2():
    print(Dashes + "Starting TestKerasSegmentation2" + Dashes)
    results = "Results Keras Segmentation 2"
    CreateDirs(results, GT=True)

    dim = 256
    minus = 0
    BatchSize = 8
    CropPerImage = 1

    inputs = "/Users/firetiti/NN/DataSets/Test Inputs Gray/"
    outputs = "/Users/firetiti/NN/DataSets/Test Outputs/"

    generator = ImageDataGenerator.Generator(ChannelFirst=True)
    generator.setCropPerImage(CropPerImage)
    generator.setShuffle(True)
    generator.setFlip(True)
    generator.setRotate90x(True)
    generator.setMaxShiftRange(1000)
    generator.setNoise(31, "Uniform", 51)
    generator.setBrighterDarker(53, "Uniform_PerChannel")
    generator.LoadInputs(inputs, OnTheFly=False, Classification=False)
    generator.LoadOutputs(outputs)
    generator.setInputsDimensions(dim, dim)
    generator.setOutputsDimensions(dim - minus, dim - minus)
    gen = generator.Keras(BatchSize, InputsNormalizers=None, OutputsNormalizers=None)

    nbEpoch = 3
    for epoch in range(nbEpoch):
        for batch in range(gen.getNbBatchPerEpoch()):
            X, Y = gen.next()

            prefix = str(epoch) + " - Batch " + str(batch) + " - "
            for i in range(X.shape[0]):
                Processing.ForceRange(X[i])
                ImagesIO.Write(X[i][0], True, results + "/In/Epoch " + prefix + str(i) + " - Input.png")
                ImagesIO.Write(Y[i][0], True, results + "/GT/Epoch " + prefix + str(i) + " - Output.png")

    print("TestKerasSegmentation2 Done!\n\n\n\n\n")


def TestPyTorchSegmentation1():
    print(Dashes + "Starting TestPyTorchSegmentation1" + Dashes)
    results = "Results PyTorch Segmentation 1"
    CreateDirs(results, GT=True)

    dim = 256
    minus = 10
    BatchSize = 10
    CropPerImage = 2

    inputs = "/Users/firetiti/NN/DataSets/Test Segmentation Inputs/"
    outputs = "/Users/firetiti/NN/DataSets/Test Segmentation Outputs/"

    InputsNormalizers = [Normalizers.Normalize(), Normalizers.Normalize()]
    OutputsNormalizers = [Normalizers.Basic(), Normalizers.Basic()]

    generator = ImageDataGenerator.Generator(ChannelFirst=False)
    generator.setCropPerImage(CropPerImage)
    generator.setShuffle(True)
    generator.setFlip(True)
    generator.setRotate90x(True)
    # generator.setRotate(True, 90, FillingValues=[100,200])
    generator.setMaxShiftRange(1000)
    # generator.setNoise(31, "Uniform", 73)
    generator.LoadInputs(inputs, OnTheFly=False, Classification=False)
    generator.LoadOutputs(outputs)
    generator.setInputsDimensions(dim, dim)
    generator.setOutputsDimensions(dim - minus, dim - minus)
    gen = generator.PyTorch(BatchSize, InputsNormalizers=InputsNormalizers, OutputsNormalizers=OutputsNormalizers,
                            Workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device = " + str(device))

    nbEpoch = 2
    for epoch in range(nbEpoch):
        for b, batch in enumerate(gen):
            X, Y = batch['input'].to(device), batch['output'].to(device)

            for innorm, pos in zip(InputsNormalizers, range(len(InputsNormalizers))):
                innorm.Denormalize(X[:, pos, :, :, :])

            for outnorm, out in zip(OutputsNormalizers, range(len(OutputsNormalizers))):
                outnorm.Denormalize(Y[:, out, :, :, :])

            for i in range(X.shape[0]):
                index = 0
                prefix = str(epoch) + " - Batch " + str(b) + " - Pair " + str(i) + " - "
                for inputs, outputs in zip(X[i], Y[i]):
                    ImagesIO.Write(inputs.detach().cpu().squeeze().numpy(), False,
                                   results + "/In/Epoch " + prefix + str(index) + " - Input.png")
                    ImagesIO.Write(outputs.detach().cpu().squeeze().numpy(), False,
                                   results + "/GT/Epoch " + prefix + str(index) + " - Output.png")
                    index += 1

    print("TestPyTorchSegmentation1 Done!\n\n\n\n\n")


def TestPyTorchSegmentation2():
    print(Dashes + "Starting TestPyTorchSegmentation2" + Dashes)
    results = "Results PyTorch Segmentation 2"
    CreateDirs(results, GT=True)

    dim = 256
    minus = 0
    BatchSize = 8
    CropPerImage = 1

    inputs = "/Users/firetiti/NN/DataSets/Test Inputs Gray/"
    outputs = "/Users/firetiti/NN/DataSets/Test Outputs/"

    InputsNormalizers = [Normalizers.Normalize()]
    OutputsNormalizers = [Normalizers.Basic()]

    generator = ImageDataGenerator.Generator(ChannelFirst=False)
    generator.setCropPerImage(CropPerImage)
    generator.setBrighterDarker(53, "Gaussian_PerChannel")
    generator.LoadInputs(inputs, OnTheFly=True, Classification=False)
    generator.LoadOutputs(outputs)
    generator.setInputsDimensions(dim, dim)
    generator.setOutputsDimensions(dim - minus, dim - minus)
    gen = generator.PyTorch(BatchSize, InputsNormalizers=InputsNormalizers, OutputsNormalizers=OutputsNormalizers,
                            Workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Device = " + str(device))

    nbEpoch = 2
    for epoch in range(nbEpoch):
        for b, batch in enumerate(gen):
            X, Y = batch['input'].to(device), batch['output'].to(device)
            InputsNormalizers[0].Denormalize(X)
            OutputsNormalizers[0].Denormalize(Y)
            prefix = str(epoch) + " - Batch " + str(b) + " - "
            for i in range(X.shape[0]):
                Processing.ForceRange(X[i])
                ImagesIO.Write(X[i].detach().cpu().squeeze().numpy(), False,
                               results + "/In/Epoch " + prefix + str(i) + " - Input.png")
                ImagesIO.Write(Y[i].detach().cpu().squeeze().numpy(), False,
                               results + "/GT/Epoch " + prefix + str(i) + " - Output.png")

    print("TestPyTorchSegmentation2 Done!\n\n\n\n\n")


def TestPyTorchSegmentation3():
    print(Dashes + "Starting TestPyTorchSegmentation3" + Dashes)
    results = "Results PyTorch Segmentation 3"
    CreateDirs(results, GT=True)
    os.makedirs(results + "/GT/Distance Maps/", exist_ok=True)
    os.makedirs(results + "/GT/Foreground/", exist_ok=True)
    os.makedirs(results + "/GT/Originals/", exist_ok=True)
    os.makedirs(results + "/GT/Separations/", exist_ok=True)

    ## WANDB
    HPP_DEFAULT = Dotdict(dict(
        batch_size=8,
        dimensions=256,
        minus=0,
        nbCropPerImage=1
    ))
    run = wandb.init(project="firetitilib", config=HPP_DEFAULT)

    Datasets = []

    Path = "../datasets/Cyclic IF/002 - Cropped - Small/"
    inputs = Path + "/Originals Stretched/"
    outputs = Path + "/Outs 01/"
    Generator1 = ImageDataGenerator.Generator(ChannelFirst=True)
    Generator1.setShuffle(False)
    Generator1.setCropPerImage(2)
    Generator1.setMaxShiftRange(10000)
    Generator1.setKeepEmptyOutputProbability(1.0)
    Generator1.LoadInputs(inputs, OnTheFly=False, Classification=False)
    Generator1.LoadOutputs(outputs)
    Generator1.setInputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Generator1.setOutputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Dataset1 = Generator1.PyTorchDataset(10, InputsNormalizers=None, OutputsNormalizers=None)
    Datasets.append({"Dataset": Dataset1, "Length": Dataset1.__len__()})
    print("Main Dataset created.")

    Path = "../datasets/Cyclic IF/Broad/"
    inputs = Path + "/Originals Stretched/"
    outputs = Path + "/Outs 01/"
    Generator2 = ImageDataGenerator.Generator(ChannelFirst=True)
    Generator2.setShuffle(False)
    Generator2.setCropPerImage(1)
    Generator2.setMaxShiftRange(10000)
    Generator2.setKeepEmptyOutputProbability(1.0)
    Generator2.LoadInputs(inputs, OnTheFly=False, Classification=False)
    Generator2.LoadOutputs(outputs)
    Generator2.setInputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Generator2.setOutputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Dataset2 = Generator2.PyTorchDataset(2, InputsNormalizers=None, OutputsNormalizers=None)
    Datasets.append({"Dataset": Dataset2, "Length": 2})

    Path = "../datasets/Cyclic IF/DSB 2018/696x520/"
    inputs = Path + "/Originals Stretched/"
    outputs = Path + "/Outs 01/"
    Generator3 = ImageDataGenerator.Generator(ChannelFirst=True)
    Generator3.setShuffle(False)
    Generator3.setCropPerImage(1)
    Generator3.setMaxShiftRange(10000)
    Generator3.setKeepEmptyOutputProbability(0)
    Generator3.LoadInputs(inputs, OnTheFly=False, Classification=False)
    Generator3.LoadOutputs(outputs)
    Generator3.setInputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Generator3.setOutputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Dataset3 = Generator3.PyTorchDataset(1, InputsNormalizers=None, OutputsNormalizers=None)
    Datasets.append({"Dataset": Dataset3, "Length": 1})

    Path = "../datasets/Cyclic IF/DSB 2018/1272x603/"
    inputs = Path + "/Originals Stretched/"
    outputs = Path + "/Outs 01/"
    Generator4 = ImageDataGenerator.Generator(ChannelFirst=True)
    Generator4.setShuffle(False)
    Generator4.setCropPerImage(1)
    Generator4.setMaxShiftRange(10000)
    Generator4.setKeepEmptyOutputProbability(0)
    Generator4.LoadInputs(inputs, OnTheFly=False, Classification=False)
    Generator4.LoadOutputs(outputs)
    Generator4.setInputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Generator4.setOutputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Dataset4 = Generator4.PyTorchDataset(1, InputsNormalizers=None, OutputsNormalizers=None)
    Datasets.append({"Dataset": Dataset4, "Length": 2})

    Path = "../datasets/Cyclic IF/S-BSST265/"
    inputs = Path + "/Originals Stretched/"
    outputs = Path + "/Outs 01/"
    Generator5 = ImageDataGenerator.Generator(ChannelFirst=True)
    Generator5.setShuffle(False)
    Generator5.setCropPerImage(1)
    Generator5.setMaxShiftRange(10000)
    Generator5.setKeepEmptyOutputProbability(0)
    Generator5.LoadInputs(inputs, OnTheFly=False, Classification=False)
    Generator5.LoadOutputs(outputs)
    Generator5.setInputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Generator5.setOutputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    Dataset5 = Generator5.PyTorchDataset(1, InputsNormalizers=None, OutputsNormalizers=None)
    Datasets.append({"Dataset": Dataset5, "Length": 1})

    SuperDataset = ImageDataGenerator.PyTorchMultipleDatasets(Datasets, HPP_DEFAULT.batch_size)
    print("Super Dataset created!!!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warning = "" if str(device) == "cuda" else "WARNING - "
    print(warning + "Device = " + str(device))

    gen = DataLoader(SuperDataset, batch_size=HPP_DEFAULT.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    nbEpoch = 2
    for epoch in range(nbEpoch):
        for b, batch in enumerate(gen):
            X, Y = batch['input'].to(device), batch['output'].to(device)

            prefix = str(epoch) + " - Batch " + str(b) + " - "
            for i in range(X.shape[0]):
                Processing.ForceRange(X[i], max=65535.0 if i < 10 else 255.0)
                ImagesIO.Write(X[i].detach().cpu().squeeze().numpy(), True,
                               results + "/In/Epoch " + prefix + str(i) + " - Input.png")
                ImagesIO.Write(Y[i][0].detach().cpu().squeeze().numpy(), True,
                               results + "/GT/Distance Maps/Epoch " + prefix + str(i) + " - Out 0.tif",
                               FloatEncoding=True)
                ImagesIO.Write(Y[i][1].detach().cpu().squeeze().numpy(), True,
                               results + "/GT/Foreground/Epoch " + prefix + str(i) + " - Out 1.png")
                ImagesIO.Write(Y[i][2].detach().cpu().squeeze().numpy(), True,
                               results + "/GT/Originals/Epoch " + prefix + str(i) + " - Out 2.png")
                ImagesIO.Write(Y[i][3].detach().cpu().squeeze().numpy(), True,
                               results + "/GT/Separations/Epoch " + prefix + str(i) + " - Out 3.png")

    print("TestPyTorchSegmentation3 Part 1 Done!\n\n")

    model = ResUnetA.ResUnetA(Inputs=1, Depth=3, FeatureMaps=4, Activations="relu",
                              ResidualConnection="conv", FirstLastBlock="ResBlock", Dilations=(1, 13, 12),
                              DownSampling="max_pool", UpSampling="nearest", PSPpooling=[1, 3, 7],
                              BatchNormEncoder=[True, True, True], InstNormDecoder=[True, True, True],
                              Attention=['EA_64_1_64_BN_0', 'EA_64_1_64_IN_0', 'NLE_None_2_True_True'],
                              DropOutPosition="Block", DropOut=None,
                              OutputActivations=["sigmoid", "sigmoid", "sigmoid", "sigmoid"], ConcatenateOutputs=True)
    model = model.to(device)
    # loq_freq shouldn't be 1, but needed here because of the very low number of images
    wandb.watch(model, log="all", log_freq=1)

    loss_fn = Losses.MultipleOutputs([Losses.DiceLoss_Tanimoto_Dual(), Losses.DiceLoss_Tanimoto_Dual(),
                                      Losses.DiceLoss_Tanimoto_Dual(), Losses.DiceLoss_Tanimoto_Dual()])

    optimizer = torch.optim.Adam(model.parameters())

    inputs = "./Results PyTorch Segmentation 3/In/"
    outputs = "./Results PyTorch Segmentation 3/GT/"

    InputsNormalizers = [Normalizers.CenterReduce()]
    OutputsNormalizers = [Normalizers.Basic(MaxValue=30.0), Normalizers.Basic(MaxValue=255.0),
                          Normalizers.Basic(MaxValue=65535.0), Normalizers.Basic(MaxValue=255.0)]

    generator = ImageDataGenerator.Generator(ChannelFirst=True)
    generator.setShuffle(False)
    generator.setCropPerImage(HPP_DEFAULT.nbCropPerImage)
    generator.setFlip(True)
    generator.setRotate90x(True)
    generator.setKeepEmptyOutputProbability(0.73)
    generator.LoadInputs(inputs, OnTheFly=False, Classification=False)
    generator.LoadOutputs(outputs)
    generator.setInputsDimensions(HPP_DEFAULT.dimensions, HPP_DEFAULT.dimensions)
    generator.setOutputsDimensions(HPP_DEFAULT.dimensions - HPP_DEFAULT.minus,
                                   HPP_DEFAULT.dimensions - HPP_DEFAULT.minus)
    dl = generator.PyTorch(HPP_DEFAULT.batch_size, InputsNormalizers=InputsNormalizers,
                           OutputsNormalizers=OutputsNormalizers,
                           Workers=0)

    losshistory = []
    nbepochs = 3
    for epoch in range(nbepochs):
        lossvalue = 0.0
        start = time.time()
        for b, batch in enumerate(dl):
            X, Y = batch["input"].to(device, dtype=torch.float), batch["output"].to(device, dtype=torch.float)

            optimizer.zero_grad()  # Zero the gradients before running the backward pass.

            Ypred = model(X)  # Forward pass: compute predicted y

            loss = loss_fn(Ypred, Y)  # Compute and print loss
            lossvalue += loss.item()
            loss.backward()

            optimizer.step()
        wandb.log({
            "loss": lossvalue / (b + 1)
        })
        end = time.time()
        print("Epoch %d - loss = %f - %f s" % (epoch, lossvalue, (end - start)))
        losshistory.append(lossvalue)
    print("Training Done!\n")

    print("TestPyTorchSegmentation3 Part 2 Done!\n\n")

    print("TestPyTorchSegmentation3 Done!\n\n\n\n\n")


def TestImageIOandDimensions():
    print(Dashes + "Running TestImageIOandDimensions" + Dashes)
    Errors = 0

    Tests = [{"path": "/Users/firetiti/NetBeans/Images/Colors/JCBRUET_9.jpg", "dim": (330, 500, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Lena unpublished.jpg", "dim": (292, 433, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/harlequin.png", "dim": (318, 270, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/flower.png", "dim": (648, 486, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Dasha Astafieva 02.png", "dim": (890, 1127, 4)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Dasha Astafieva 04.png", "dim": (1502, 1002, 4)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Noise - f9a70047563873.587e548dd0302.jpeg",
              "dim": (1400, 2100, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Lena fullhd.jpg", "dim": (1084, 2318, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/foreman_color.tif", "dim": (174, 144, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Skins/Moi 072.JPG", "dim": (2448, 3264, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Skins/DouDou & Moi 127.JPG", "dim": (2448, 3264, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/Anno_C820 KC 79days 0.png", "dim": (256, 200, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Colors/CHENLE_1.jpg", "dim": (500, 333, 3)},
             {"path": "/Users/firetiti/NetBeans/Images/Gray Levels/Dasha Astafieva 05.png", "dim": (885, 1127, 1)},
             {
                 "path": "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/Originals/BR1506-A015 - Scene 017.png",
                 "dim": (7558, 7560, 1)},
             {
                 "path": "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/Originals/BR1506-A015 - Scene 049.tif",
                 "dim": (5720, 5726, 1)},
             {
                 "path": "/Users/firetiti/Downloads/CyclicIF/IF Manual Segmentation/BR1506-A015/Originals/BR1506-A015 - Scene 059.tif",
                 "dim": (7561, 5728, 1)}]

    path = []
    for test in Tests:  # Test 0
        try:
            path.append(test["path"])

            image = imageio.imread(test["path"])

            width, height, channels, first = ImageTools.Dimensions(image)

            if width != test["dim"][0] or height != test["dim"][1] or channels != test["dim"][2]:
                print("Error 0 - " + test["path"] + " => Shape=" + str(image.shape)
                      + ", Dim=(" + str(width) + "," + str(height) + "," + str(channels) + "," + str(
                    first) + "), GT=" + str(test["dim"]))
                Errors += 1

        except IOError as e:
            print("Could not read '%s': %s - it\'s ok, skipping." % (test["path"], e))
            sys.stdout.flush()
            Errors += 1
    print("Subtest 1 done.")

    Images = ImagesIO.LoadImagesList(path, False, ReturnImagesList=False, verbose=False)  # Test 1
    for image, test in zip(Images, Tests):
        try:
            width, height, channels, first = ImageTools.Dimensions(image)

            if width != test["dim"][0] or height != test["dim"][1] or channels != test["dim"][2] or (
                    1 < channels and first != False):
                print("Error 1 - " + test["path"] + " => Shape=" + str(image.shape)
                      + ", Dim=(" + str(width) + "," + str(height) + "," + str(channels) + "," + str(
                    first) + "), GT=" + str(test["dim"]))
                Errors += 1

        except IOError as e:
            print("Could not read '%s': %s - it\'s ok, skipping." % (test["path"], e))
            sys.stdout.flush()
            Errors += 1
    print("Subtest 2 done.")

    Images = ImagesIO.LoadImagesList(path, True, ReturnImagesList=False, verbose=False)  # Test 2
    for image, test in zip(Images, Tests):
        try:
            width, height, channels, first = ImageTools.Dimensions(image)

            if width != test["dim"][0] or height != test["dim"][1] or channels != test["dim"][2] or (
                    1 < channels and first != True):
                print("Error 2 - " + test["path"] + " => Shape=" + str(image.shape)
                      + ", Dim=(" + str(width) + "," + str(height) + "," + str(channels) + "," + str(
                    first) + "), GT=" + str(test["dim"]))
                Errors += 1

        except IOError as e:
            print("Could not read '%s': %s - it\'s ok, skipping." % (test["path"], e))
            sys.stdout.flush()
            Errors += 1
    print("Subtest 3 done.")

    print(
        "TestImageIOandDimensions done with " + str(Errors) + " error" + str("" if Errors <= 1 else "s") + "\n\n\n\n\n")


def main():
    # TestKerasClassification1()
    # TestKerasClassification2()
    # TestKerasSegmentation1()
    # TestKerasSegmentation2()

    # TestPyTorchSegmentation1()
    # TestPyTorchSegmentation2()
    TestPyTorchSegmentation3()

    # TestImageIOandDimensions()

    print("All Done!")
    sys.exit(0)


if __name__ == "__main__":
    main()
