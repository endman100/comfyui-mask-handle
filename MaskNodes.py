import os
import cv2
import torch
import shutil
from datetime import datetime
import numpy as np
import time
from collections.abc import Iterable

 
class MaskSelectionByIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask, index=0):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        if index >= len(mask):
            raise ValueError("index out of range", index, len(mask))
            

        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            result = mask[index].unsqueeze(0)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            result = [mask[index]]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (result,) # 返回结果 shape 为 (1, h, w)
    
class MaskSelectionMaxSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            min_size = 0
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size > min_size:
                    min_size = _mask.size
                    min_index = index
            result = mask[min_index].unsqueeze(0)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            min_size = 0
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size > min_size:
                    min_size = _mask.size
                    min_index = index
            result = [mask[min_index]]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (result,)
    
class MaskSelectionMinSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            min_size = np.inf
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size < min_size:
                    min_size = _mask.size
                    min_index = index
            result = mask[min_index].unsqueeze(0)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            min_size = np.inf
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size < min_size:
                    min_size = _mask.size
                    min_index = index
            result = [mask[min_index]]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (result,)
class MaskSelectionMaxCountours:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        elif(len(mask) == 0):
            return mask
        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            max_countour = 0
            max_countour_area = 0
            max_index = 0
            for index, _mask in enumerate(mask):
                _, countours, _ = cv2.findContours(_mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for countour in countours:
                    area = cv2.contourArea(countour)
                    if area > max_countour_area:
                        max_countour_area = area
                        max_countour = countour
                        max_index = index
            print(mask[max_index])
            h, w = mask[0].shape[1], mask[0].shape[2]
            result = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(result, max_countour, -1, 255, -1)
            result = torch.tensor(result).unsqueeze(0)
            print(result)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            max_countour = 0
            max_countour_area = 0
            max_index = 0
            for index, _mask in enumerate(mask):
                _, countours, _ = cv2.findContours(_mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for countour in countours:
                    area = cv2.contourArea(countour)
                    if area > max_countour_area:
                        max_countour_area = area
                        max_countour = countour
                        max_index = index
            h, w = mask[0].shape[1], mask[0].shape[2]
            result = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(result, max_countour, -1, 255, -1)
            result = torch.tensor(result).unsqueeze(0)
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (result,)

class MaskOrMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask1, mask2, merge_all):
        if (isinstance(mask1, torch.Tensor) and mask1.dim() == 3 and 
            isinstance(mask2, torch.Tensor) and mask2.dim() == 3):
            original_dtype = mask1.dtype
            mask1 = mask1.bool()
            mask2 = mask2.bool()
            result = torch.logical_or(mask1, mask2)
            if(merge_all):
                result = torch.any(result, dim=0)
            result = result.to(original_dtype)

        elif isinstance(mask1, list) and isinstance(mask2, list):
            result = []
            for _mask1, _mask2 in zip(mask1, mask2):
                original_dtype = _mask1.dtype
                _mask1 = _mask1.bool()
                _mask2 = _mask2.bool()
                _result = torch.logical_or(_mask1, _mask2)
                result.append(_result)

            if(merge_all):
                result = [torch.any(_result, dim=0) for _result in result]
            result = [ _result.to(original_dtype) for _result in result]            
        else:
            raise ValueError("mask is not list or tensor", mask1.shape, mask2.shape, type(mask1), type(mask2))
        return (result,)

class MaskAndMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask1, mask2, merge_all):
        if (isinstance(mask1, torch.Tensor) and mask1.dim() == 3 and 
            isinstance(mask2, torch.Tensor) and mask2.dim() == 3):
            original_dtype = mask1.dtype
            mask1 = mask1.bool()
            mask2 = mask2.bool()
            result = torch.logical_and(mask1, mask2)
            if(merge_all):
                result = torch.all(result, dim=0)
            result = result.to(original_dtype)

        elif isinstance(mask1, list) and isinstance(mask2, list):
            result = []
            for _mask1, _mask2 in zip(mask1, mask2):
                original_dtype = _mask1.dtype
                _mask1 = _mask1.bool()
                _mask2 = _mask2.bool()
                _result = torch.logical_and(_mask1, _mask2)
                result.append(_result)

            if(merge_all):
                result = [torch.all(_result, dim=0) for _result in result]
            result = [ _result.to(original_dtype) for _result in result]            
        else:
            raise ValueError("mask is not list or tensor", mask1.shape, mask2.shape, type(mask1), type(mask2))
        return (result,)
class MaskSubMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask1, mask2, merge_all):
        if (isinstance(mask1, torch.Tensor) and mask1.dim() == 3 and 
            isinstance(mask2, torch.Tensor) and mask2.dim() == 3):
            original_dtype = mask1.dtype
            mask1 = mask1.bool()
            mask2 = mask2.bool()
            result = torch.logical_and(mask1, ~mask2)
            if(merge_all):
                result = torch.all(result, dim=0)
            result = result.to(original_dtype)

        elif isinstance(mask1, list) and isinstance(mask2, list):
            result = []
            for _mask1, _mask2 in zip(mask1, mask2):
                original_dtype = _mask1.dtype
                _mask1 = _mask1.bool()
                _mask2 = _mask2.bool()
                _result = torch.logical_and(_mask1, ~_mask2)
                result.append(_result)

            if(merge_all):
                result = [torch.all(_result, dim=0) for _result in result]
            result = [ _result.to(original_dtype) for _result in result]            
        else:
            raise ValueError("mask is not list or tensor", mask1.shape, mask2.shape, type(mask1), type(mask2))
        return (result,)

class MaskInvert:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask):
        if isinstance(mask, torch.Tensor) and mask.dim() == 3:
            original_dtype = mask.dtype
            mask = mask.bool()
            result = ~mask
            result = result.to(original_dtype)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            original_dtype = mask.dtype
            mask = mask.bool()
            result = ~mask
            result = result.unsqueeze(0).to(original_dtype)
        elif isinstance(mask, list):
            result = []
            for _mask in mask:
                original_dtype = _mask.dtype
                _mask = _mask.bool()
                _result = ~_mask
                result.append(_result)
            result = [ _result.to(original_dtype) for _result in result]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (result,)

NODE_CLASS_MAPPINGS = {
    "Mask Selection By Index (endman100)": MaskSelectionByIndex,
    "Mask Selection Max Size (endman100)": MaskSelectionMaxSize,
    "Mask Selection Min Size (endman100)": MaskSelectionMinSize,
    "Mask Selection Max Countours (endman100)": MaskSelectionMaxCountours,
    "Mask Or Mask (endman100)": MaskOrMask,
    "Mask And Mask (endman100)": MaskAndMask,
    "Mask Sub Mask (endman100)": MaskSubMask,
    "Mask Invert (endman100)": MaskInvert
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask Selection By Index (endman100)": "Mask Selection By Index (endman100)",
    "Mask Selection Max Size (endman100)": "Mask Selection Max Size (endman100)",
    "Mask Selection Min Size (endman100)": "Mask Selection Min Size (endman100)",
    "Mask Selection Max Countours (endman100)": "Mask Selection Max Countours (endman100)",
    "Mask Or Mask (endman100)": "Mask Or Mask (endman100)",
    "Mask And Mask (endman100)": "Mask And Mask (endman100)",
    "Mask Sub Mask (endman100)": "Mask Sub Mask (endman100)",
    "Mask Invert (endman100)": "Mask Invert (endman100)"
}
