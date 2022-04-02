# Classification_Gochiusa
"Is your order a rabbit?" Classification of characters in.  

## Classes(Character)  
 - Chino  
 - Cocoa  

## Versions
 - Docker version 20.10.8, build 3967b7d  
 - docker-compose version 1.29.2, build 5becea4c   
 - Python 3.7.13  
 - numpy 1.21.5  
 - tensorflow 1.14.0  
 - keras 2.2.5  
 - h5py 2.10.0  

### Note
    Note that the library version is out of date and an error will occur.

## Run

### Docker
 - `docker compose up -d --build`
 - `docker compose exec app bash`  

 ### Crawling image
 - `python opt/craw_image.py`  
 In some cases, images cannot be collected successfully. In that case, collect images manually.  

 ### Preprocessing
 - `python opt/preprocessing.py`  

 ### Learning
 - `python opt/train.py`  

 ### Classification
Set the path to the image in the codeinfo file.  
`QUESTION_PIC = "opt/foo/bar.jpg"`  

 - `python opt/main.py`  