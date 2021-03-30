# Principal component analysis (PCA) 🔥
This is a showcase on how PCA is used for dimensionality reduction. Putting aside non-essential dimensions, especially for the case of big datasets, helps programmers and data scientists to work with data in a more efficient way.  
Employing PCA, this code tries to find imortant principal components of images provided in `Database` as a whole and by omitting insignificant features, it attempts to reduce its size.  

# Usage 🛠️
✔️ After installing required python libraries by entering
`pip install -r requirements.txt`
into a terminal, the program can easily start with 
`python PCA_Showcase.py`.  
✔️ Also you can manually pick out an arbirtrary number of Eigen vectors to get more insight into PCA.  

# Output 🗜️
For images in `Database`, as much as of a tenth of dataset is thrown away, however, after reconstructing images out of the new dataset no discernible difference between them is found.