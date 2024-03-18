model: Zernike-3 (Z-3)

TODO
1. wrap notebook code into a script/module
2. write test suite (diff image input size/shape, output shape, etc)
3. research how to convert a1, a3 calculation into distance
    - acquire calibration image stack


user inputs
- (x,y) of position to focus on
- patch size: changes resolution of focuser
- step size: changes overlap of focuser

system needs to be able to handle:
- diff sized inputs


===========

autofocus algorithm:
1. build calibration curve of blurriness scores vs distance
    1. acquire a stack of images
    2. calculate a_1 for each (blurriness score)
2. test of sample image
    1. calculate PSF map and distance map 