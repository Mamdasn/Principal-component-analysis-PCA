import numpy as np
import os
import platform
import cv2
from time import sleep

slash = '\\' if platform.system() == 'Windows' else '/'

Destination  = f'.{slash}Output'
Input_images = f'.{slash}Database'
if not os.path.exists(Destination):
    os.makedirs(Destination)

def image2double(img):
    if (img.dtype != np.uint8):
        raise ValueError("ValueError exception thrown!\n \
                         \r\tInput img should be of dtype uint8.")
    return img.astype(np.float) / (np.iinfo(img.dtype).max - np.iinfo(img.dtype).min)
def image2uint8(img):
    if (img.dtype != np.float):
        raise ValueError("ValueError exception thrown!\n \
                         \r\tInput img should be of dtype float.")
    return (255*img).astype(np.uint8)

def display_images(img_fullname, img, img_reconstructed, psnr):
    h, w =  img.shape
    if (w < 250):
        h = h * 250//w 
        w = 250
        img = cv2.resize(img, dsize=(h, w))
        img_reconstructed = cv2.resize(img_reconstructed, dsize=(h, w))
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_img = np.ones((30, 2*w+90))
    title_img =  cv2.putText(title_img, 'Original Image', (80, 20), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    title_img_names = cv2.putText(title_img, 'Reconstructed Image', (2*w+90-245, 20), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    title_img_psnr = cv2.putText(np.ones((30, 2*w+90)), f'PSNR: {psnr}', (20, 20), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    img_or = np.concatenate((
                    np.concatenate((np.ones((h, 20)), img), axis=1), 
                    np.ones((h, 50)), 
                    np.concatenate((img_reconstructed, np.ones((h, 20))), axis=1)
                    ), 
                    axis=1
                    )
    
    img_or_t = np.concatenate((
                    title_img_names,
                    img_or,
                    title_img_psnr
                ),
                axis=0
            )
    
    # cv2.imwrite(f'Figures{slash}Figure-{img_fullname}', image2uint8(img_or_t))
    cv2.imshow('Original Image - Reconstructed Image', img_or_t)
    cv2.waitKey(1000)
    cv2.destroyAllWindows() 

def main():
    N = 160
    print('Creating Database')
    DB = []
    DB_img_sizes = []
    img_fullnames = os.listdir(Input_images)
    for img_fullname in img_fullnames:
        img_itr = cv2.imread(f'{Input_images}{slash}{img_fullname}', cv2.IMREAD_GRAYSCALE)
        DB_img_sizes.append(img_itr.shape)
        img_itr = image2double(cv2.resize(img_itr, dsize=(N, N)))
        DB.append(img_itr.T.reshape((-1), ))
    DB = np.array(DB)
    DB_mean = np.mean(DB, axis=0) # calculating mean of columns in DB
    
    # Zero-ing mean of DB
    DB = DB - DB_mean
    
    print('Calculating Covariance')
    DB_cv = np.cov(DB)
    print('DB_cv.shape:', DB_cv.shape)
    print('Calculating Eigen Vector and Eigen Values')
    [eigenValues, eigenVectors] = np.linalg.eigh(DB_cv)

    sort_index     = eigenValues.argsort()[::-1] # Descending Order
    eigenValues    = eigenValues[sort_index]
    eigenVectors   = eigenVectors[:, sort_index]
    for indx in range(len(eigenValues)-1):
        if abs(eigenValues[indx]) > abs(1e5 * eigenValues[indx+1]):
            featureVectors = eigenVectors[:, 0:indx+1]
            featureValues  = eigenValues[0:indx+1]
            break
    else:
        featureVectors = eigenVectors
        featureValues  = eigenValues
    # or manually pick p out of n dimenstions
    # featureVectors = eigenVectors[:, 0:5]
    
    print('featureValues:', featureValues)
    
    print('featureVectors.shape:', featureVectors.shape)
    print('DB.shape:', DB.shape)

    DB_PCAed = np.matmul(featureVectors.T, DB)
    print('DB_PCAed.shape', DB_PCAed.shape)
    
    print("Reconstructing Database")
    DB_reconstructed = np.matmul(featureVectors, DB_PCAed) + DB_mean
    print('DB_reconstructed.shape', DB_reconstructed.shape)
    
    # deZero-ing mean of DB
    DB = DB + DB_mean

    print("Saving Images")
    for indx, img_fullname in enumerate(img_fullnames):
        img = np.reshape(DB[indx, :], (N, N)).T
        img_reconstructed = np.reshape(DB_reconstructed[indx, :], (N, N)).T
        
        h, w = DB_img_sizes[indx]
        img = cv2.resize(img, dsize=(h, w))
        img_reconstructed = cv2.resize(img_reconstructed, dsize=(h, w))
        psnr = cv2.PSNR(img, img_reconstructed)
        print(img_fullname, '\tpsnr: ', psnr)
        
        display_images(img_fullname, img, img_reconstructed, round(psnr, 2))
        
        img_name, img_extension = os.path.splitext(img_fullname)
        # cv2.imwrite(f"{Destination}{slash}{img_name}{img_extension}", image2uint8(img))
        cv2.imwrite(f"{Destination}{slash}{img_name}_reconstructed_{round(psnr, 2)}{img_extension}", image2uint8(img_reconstructed))
        

if __name__ == '__main__': main()



    
    







    
    
