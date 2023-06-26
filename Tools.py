import numpy as np
import ntpath
import os
import sys
import traceback


# ----------------------------------------------------- Misc -----------------------------------------------------


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# ----------------------------------------------------- Shuffle -----------------------------------------------------
def Permute2(a, b, index):
    tmp = a[index]
    a[index] = b[index]
    b[index] = tmp


def Permute(a, b):
    tmp = a
    a = b
    b = tmp


def Shuffle(a, verbose=False):
    """ This function performs shuffling.
        Args:
            a: The dataset to shuffle.
            verbose (bool): Use verbose mode?
    """
    np.random.shuffle(a)
    if (verbose):
        print("Shuffling done")
        sys.stdout.flush()


def ShuffleUnison(a, b, verbose=False):
    """ This function performs the exact same shuffling on two datasets of images.
        Args:
            a: The first dataset to shuffle.
            b: The second dataset to shuffle.
            verbose (bool): Use verbose mode?
    """
    assert len(a) == len(b)
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    if (verbose):
        print("Shuffling in Unison done")
        sys.stdout.flush()


def ShuffleUnisonXY(X: list, Y: list, verbose: bool = True):
    """ This function performs the exact same shuffling of two lists of data.
        Args:
            X (list): The first list to shuffle.
            Y (list): The second list to shuffle.
            verbose (bool): Use verbose mode?
    """
    state = np.random.get_state()

    np.random.shuffle(X[0])
    for x in range(1, len(X)):
        np.random.set_state(state)
        np.random.shuffle(X[x])

    for y in Y:
        np.random.set_state(state)
        np.random.shuffle(y)

    if (verbose):
        print("Datasets shuffling in Unison done.")
        sys.stdout.flush()


def ShuffleMultipleUnison(a, b, verbose=True):
    assert len(a) == len(b[0])
    state = np.random.get_state()
    np.random.shuffle(a)
    for dir in b:
        np.random.set_state(state)
        np.random.shuffle(dir)
    if (verbose):
        print("Multiple shuffling in Unison done")
        sys.stdout.flush()


# ----------------------------------------------------- Current path -----------------------------------------------------
def PathLeaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def PrintException(ex):
    ex_type, ex_value, ex_traceback = sys.exc_info()  # Get current system exception
    trace_back = traceback.extract_tb(ex_traceback)  # Extract unformatter stack traces as tuples
    stack_trace = list()  # Format stacktrace

    for trace in trace_back:
        stack_trace.append(
            " - File: %s; Line: %d; Func.Name: %s; Message: %s" % (trace[0], trace[1], trace[2], trace[3]))

    return "Exception type: " + str(ex_type.__name__), "Exception message: " + str(ex_value), "Stack trace: " + str(
        stack_trace)
