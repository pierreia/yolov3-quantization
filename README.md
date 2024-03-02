# YOLOv3 1.58 bits quantization  
Small experiment with quantization on YOLO v7 model.  
  
This example takes n convolutional layers randomly from a **YOLO v3** model, quantizes their weights and compare the results from regular weights.  
  

Inspired from the recent paper **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** , I made this mock  
example on YOLO v3 model. Based on **Jeff Prosise** code: https://github.com/jeffprosise/Deep-Learning/blob/master/Object%20Detection%20(YOLO).ipynb .  
  
You can download the YOLO v3 weights **coco_yolo3.h5** here: https://jeffpro.blob.core.windows.net/public/coco_yolo3.h5 .  
  
 **Needed packages**:

 `pip install keras numpy matplotlib`
 
**Usage:** 

    python3 example.py


## Example

### Before quantization

![Before_qt](before_qt.png, "Before quantization")

### After quantization

![After_qt](after_qt.png, "After quantization")

*Disclaimer*: This is not the result of any research of any kind, the purpose of this example is to see how much information  
a detection model loses when we quantize their weights in "1.58" bits. Also, there is a small "hack" in this example:  
the weights are not encoded in {-1, 0, 1} but in {-threshold, 0, threshold} where threshold is a float.