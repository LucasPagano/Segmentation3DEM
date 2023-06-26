import numpy
import sys

import torch
import torch.nn.functional as F

from torchvision.transforms import functional as F


class MaskRCNN(object):

    def __init__(self, Generator, BatchSize: int, InputsNormalizers=None, EmptyOutput: bool = False,
                 MinimumBoxSize: int = 11, Safety: int = 131):

        self.BatchSize = BatchSize
        self.EmptyOutput = EmptyOutput

        self.Safety = Safety

        self.MinimumBoxSize = MinimumBoxSize

        self.Dataset = Generator.PyTorchDataset(BatchSize, InputsNormalizers=InputsNormalizers, OutputsNormalizers=None)
        self.Item = self.Dataset.__getitem__

    def __getitem__(self, idx):
        count = 0
        while True:
            data = self.Item(idx)
            inputs, outputs = data['input'], data['output']

            if inputs.shape[0] == 1:
                inputs = numpy.tile(inputs, (3, 1, 1))
            else:
                inputs = inputs.squeeze()
            inputs = torch.as_tensor(inputs, dtype=torch.float32)

            obj_ids = numpy.unique(outputs)  # Nuclei are encoded as different labels.
            obj_ids = obj_ids[1:]  # Remove 0 which is the background.

            masks = outputs == obj_ids[:, None, None]  # Split the color-encoded mask into a set of binary masks

            nbObjects = len(obj_ids)
            if self.EmptyOutput == False and nbObjects == 0:  # Should not happen because handled by the generator.
                raise Exception("Number of objects equal 0. Not supported (yet)!")

            delete = []  # get bounding box coordinates for each mask
            boxes = []
            for i in range(nbObjects):
                pos = numpy.where(masks[i])
                xmin = numpy.min(pos[1])
                xmax = numpy.max(pos[1])
                ymin = numpy.min(pos[0])
                ymax = numpy.max(pos[0])
                if xmax < xmin + self.MinimumBoxSize or ymax < ymin + self.MinimumBoxSize:  # Remove small boxes.
                    delete.append(i)
                else:
                    boxes.append([xmin, ymin, xmax, ymax])

            if self.EmptyOutput == True or 0 < len(boxes):
                boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Convert everything into a torch.Tensor
                labels = torch.ones((nbObjects - len(delete),), dtype=torch.int64)  # There is only one class
                if 0 < len(delete):
                    masks = numpy.delete(masks, delete, axis=0)
                masks = torch.as_tensor(masks, dtype=torch.uint8)

                image_id = torch.tensor([idx])

                isSomething = torch.zeros((nbObjects - len(delete),),
                                          dtype=torch.int64)  # Suppose all instances are not objects

                if nbObjects - len(delete) == 0:  # Specific format required for no boxes.
                    boxes = boxes.reshape(-1, 4)
                    area = torch.as_tensor(0, dtype=torch.float32)
                else:
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

                Target = {}
                Target["boxes"] = boxes
                Target["labels"] = labels
                Target["masks"] = masks
                Target["image_id"] = image_id
                Target["area"] = area
                Target["isSomething"] = isSomething

                return inputs, Target

            count += 1
            if self.Safety <= count:
                raise Exception("Impossible to produce a non empty output for image index " + str(idx))

    def __len__(self):
        return self.Dataset.__len__()


class SuperMaskRCNN(object):

    def __init__(self, Dataset: int, EmptyOutput: bool = False, MinimumBoxSize: int = 11, Safety: int = 131):

        self.EmptyOutput = EmptyOutput

        self.Safety = Safety

        self.MinimumBoxSize = MinimumBoxSize

        self.Dataset = Dataset
        self.Item = self.Dataset.__getitem__

    def __getitem__(self, idx):
        count = 0
        while True:
            data = self.Item(idx)
            inputs, outputs = data['input'], data['output']

            if inputs.shape[0] == 1:
                inputs = numpy.tile(inputs, (3, 1, 1))
            else:
                inputs = inputs.squeeze()
            inputs = torch.as_tensor(inputs, dtype=torch.float32)

            obj_ids = numpy.unique(outputs)  # Nuclei are encoded as different labels.
            obj_ids = obj_ids[1:]  # Remove 0 which is the background.

            masks = outputs == obj_ids[:, None, None]  # Split the color-encoded mask into a set of binary masks

            nbObjects = len(obj_ids)
            if self.EmptyOutput == False and nbObjects == 0:  # Should not happen because handled by the generator.
                raise Exception("Number of objects equal 0. Not supported (yet)!")

            delete = []  # get bounding box coordinates for each mask
            boxes = []
            for i in range(nbObjects):
                pos = numpy.where(masks[i])
                xmin = numpy.min(pos[1])
                xmax = numpy.max(pos[1])
                ymin = numpy.min(pos[0])
                ymax = numpy.max(pos[0])
                if xmax < xmin + self.MinimumBoxSize or ymax < ymin + self.MinimumBoxSize:  # Remove small boxes.
                    delete.append(i)
                else:
                    boxes.append([xmin, ymin, xmax, ymax])

            if self.EmptyOutput == True or 0 < len(boxes):
                boxes = torch.as_tensor(boxes, dtype=torch.float32)  # Convert everything into a torch.Tensor
                labels = torch.ones((nbObjects - len(delete),), dtype=torch.int64)  # There is only one class
                if 0 < len(delete):
                    masks = numpy.delete(masks, delete, axis=0)
                masks = torch.as_tensor(masks, dtype=torch.uint8)

                image_id = torch.tensor([idx])

                isSomething = torch.zeros((nbObjects - len(delete),),
                                          dtype=torch.int64)  # Suppose all instances are not objects

                if nbObjects - len(delete) == 0:  # Specific format required for no boxes.
                    boxes = boxes.reshape(-1, 4)
                    area = torch.as_tensor(0, dtype=torch.float32)
                else:
                    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

                Target = {}
                Target["boxes"] = boxes
                Target["labels"] = labels
                Target["masks"] = masks
                Target["image_id"] = image_id
                Target["area"] = area
                Target["isSomething"] = isSomething

                return inputs, Target

            count += 1
            if self.Safety <= count:
                raise Exception("Impossible to produce a non empty output for image index " + str(idx))

    def __len__(self):
        return self.Dataset.__len__()
