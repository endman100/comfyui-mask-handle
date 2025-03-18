# comfyui-mask-handle

## Overview
comfyui-mask-handle is a collection of custom nodes designed for ComfyUI, a node-based interface for Stable Diffusion. These nodes provide advanced mask manipulation and image processing capabilities, enabling users to efficiently handle masks in various workflows. The package includes tools for selecting, combining, inverting, and transforming masks, as well as integrating them with images for tasks like area filling and transparency control.

## Objectives
The primary goal of comfyui-mask-handle is to enhance the flexibility and precision of mask-related operations within ComfyUI. It aims to:
- Simplify the selection of specific masks from batches based on index, size, or contour features.
- Enable logical operations (OR, AND, subtraction) between masks for complex region handling.
- Provide utilities for mask inversion, area filling, and alpha channel integration.
- Offer region extraction tools to identify and visualize key areas within masks.
These nodes are designed to be intuitive, efficient, and compatible with ComfyUI's workflow system, catering to both novice and advanced users.

## Mask Selection By Index (endman100)

### Description
This node selects a specific mask from a batch of masks based on the provided index. It is useful for isolating a single mask from a collection for further processing.

### Inputs
- mask (MASK): A batch of masks, which can be a list of masks or a tensor with a batch dimension.
- index (INT): The index of the mask to select, default is 0, range is 0 to 99.

### Outputs
- mask (MASK): The selected mask from the batch.

## Mask Selection Max Size (endman100)

### Description
This node selects the mask with the largest area (sum of pixel values) from a batch of masks. It is ideal for identifying the most prominent mask in a set.

### Inputs
- mask (MASK): A batch of masks.

### Outputs
- mask (MASK): The mask with the largest area.

## Mask Selection Min Size (endman100)

### Description
This node selects the mask with the smallest area (sum of pixel values) from a batch of masks. It can be used to find the least prominent mask in a set.

### Inputs
- mask (MASK): A batch of masks.

### Outputs
- mask (MASK): The mask with the smallest area.

## Mask Selection Max Contours (endman100)

### Description
This node finds the contour with the largest area across all provided masks and creates a new mask containing only that contour. It is useful for isolating the most significant region from multiple masks.

### Inputs
- mask (MASK): A batch of masks.

### Outputs
- mask (MASK): A new mask containing only the largest contour found across all input masks.

## Mask Or Mask (endman100)

### Description
This node performs a logical OR operation on two masks. If merge_all is True, it merges all masks across the batch via an OR operation.

### Inputs
- mask1 (MASK): The first mask or batch of masks.
- mask2 (MASK): The second mask or batch of masks.
- merge_all (BOOLEAN): If True, merges all masks across the batch via an OR operation, default is False.

### Outputs
- mask (MASK): The result of the OR operation between mask1 and mask2, or a merged result if merge_all is True.

## Mask And Mask (endman100)

### Description
This node performs a logical AND operation on two masks. If merge_all is True, it merges all masks across the batch via an AND operation.

### Inputs
- mask1 (MASK): The first mask or batch of masks.
- mask2 (MASK): The second mask or batch of masks.
- merge_all (BOOLEAN): If True, merges all masks across the batch via an AND operation, default is False.

### Outputs
- mask (MASK): The result of the AND operation between mask1 and mask2, or a merged result if merge_all is True.

## Mask Sub Mask (endman100)

### Description
This node performs a logical AND operation between the first mask and the inverse of the second mask, effectively subtracting the second mask from the first. If merge_all is True, it merges all masks accordingly.

### Inputs
- mask1 (MASK): The first mask or batch of masks.
- mask2 (MASK): The second mask or batch of masks to subtract from mask1.
- merge_all (BOOLEAN): If True, merges all masks across the batch via the appropriate operation, default is False.

### Outputs
- mask (MASK): The result of subtracting mask2 from mask1, or a merged result if merge_all is True.

## Mask Invert (endman100)

### Description
This node inverts the given mask, turning off pixels that were on and vice versa.

### Inputs
- mask (MASK): The mask to invert.

### Outputs
- mask (MASK): The inverted mask.

## Fill Mask Area (endman100)

### Description
This node fills the areas defined by the mask in an image with a specified color, defined by RGB values.

### Inputs
- image (IMAGE): The input image.
- mask (MASK): The mask defining the areas to fill.
- r (INT): The red component of the fill color (0-255), default is 0.
- g (INT): The green component of the fill color (0-255), default is 0.
- b (INT): The blue component of the fill color (0-255), default is 0.

### Outputs
- image (IMAGE): The image with the specified areas filled.

## Add Mask (endman100)

### Description
This node adds a mask as an alpha channel to an image, making the image transparent based on the mask.

### Inputs
- image (IMAGE): The input image.
- mask (MASK): The mask to use as the alpha channel.

### Outputs
- image (IMAGE): The image with the mask added as an alpha channel.

## Mask To Region (endman100)

### Description
This node finds the largest contour in the mask and returns its bounding box coordinates along with a preview image showing the contour and bounding box.

### Inputs
- mask (MASK): The input mask.

### Outputs
- width (int): The width of the bounding box.
- height (int): The height of the bounding box.
- x (int): The x-coordinate of the bounding box.
- y (int): The y-coordinate of the bounding box.
- preview region image (IMAGE): An image showing the contour and its bounding box.