# AutoGradeSystem
Here are the codes of the project AutoGradeSystem. The objective is to automatically grade PCV/AIA/DIP/ exams.
## Getting started
### Python & Required Libraries
The whole code are written in Python3 and a couple of libraries are used in the code. Make sure the following
libraries are installed.
* python >= 3.6.7
* numpy >= 1.16.4
* pandas >= 0.24.2
* opencv-python >= 4.1.0
* scipy >= 1.3.0
* sklearn >= 0.21.2
### How to use
Just run the script `main.py` located on the root directory without the any parameter to test it on the our collected scans. If no error occurs, a folder called **_'./all_results/'_** will be created, where the results will be stored. Several parameters within the `main.py` need to be clarified here. 
1. `digit_recognize_on` (line 29): if **True**, then the student IDs will be recognized automatically. Otherwise the student IDs are supposed to be given with a *.csv* file. For instance an example file called `student_ids_example.csv` is provided in folder *'.inputs/'*. By default it is set to **True**. The user will be firstly required to select the Region of the Interest (ROI) that contains the digits string by dragging a rectangular. Moreover, user can left click on the image to reselect the region and press ENTER to confirm your selection. More details about the digits recognition will be showed in following section.

    ![Select ROI](/store_asserts/selectROI.gif)

    **_Select ROI_**
1.  `semi_mode_on` (line 27): if **True**, human intervention is required for each answer sheet. By default it is set to **False**. When human intervention is required, normal mode is activated and you can left click the cell to mark as cross and right click to mark the cell as non-cross. You can also press *e* to enter edit mode, which allows to map the rows and press *e* again to quit that mode. By default it is set to **False**

    ![E mode](/store_asserts/Emode.gif)

    **_Enter edit mode_**

### Digits recognization
Thanks for putting back the deadline so that we can have time for the implement of the digits recognition module. we use the pca and svm to predict the single digit. the pre-trained modules are stored in folder './trained_models/'. The data set for training the modules is [MNIST](http://yann.lecun.com/exdb/mnist/). Since its the US written style of digits is different from german style. Its quite often to see that in our test scans digit 1 is miss-classified as 7 and digit 7 is miss-classified into 2. You can check those sheets in folder './example_results/CoverSheets/'. An example taken from that folder looks like below.
    ![example](/example_results/CoverSheets/ID_14345.png)
## Paper source code
location: './Latex/' 
## Contributions 
Paper: Yang Xu 60% Lei Jiao 40%

codes: Yang Xu 50% Lei Jiao 50%
## 