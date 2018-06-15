[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

First Convolutional Neural Networks (CNN) project.
In this project, i will build a function that when given an image of a dog, it will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

## Project Instructions

### Instructions

Previously you would need to download the datasets manually however i have added a download function that gathers these as needed.
After setting up your environment using the yaml / pip you can run all the cells in the jupyter notebook. The first time it will run slowly due to the need to download the data sets and pretrained networks. Approx 5GB in total.
There are a few options in using different pretrained CNNs later in the file. I have set this up to use my default choice, (this is the larger pretrained data download) but it also achieves the best results. You are welcome to change the choice and it will download accordingly.

1. **If you are running the project on your local machine**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

2. (Optional) **If you are running the project on your local machine** and Step 1 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
3. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

4. (Optional) **If you are running the project on your local machine**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dlnd_rep_2 --display-name "dlnd_rep_2"
```

5. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

6. (Optional) **If you are running the project on your local machine**, before running code, change the kernel to match the dlnd_rep_2 environment by using the drop-down menu (**Kernel > Change kernel > dlnd_rep_2**). Then, follow the instructions in the notebook.