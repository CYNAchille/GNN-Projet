import numpy as np
from scipy.spatial.distance import cdist
import torch
from skimage import transform
import scipy.ndimage

def sparsify_graph(A, knn_graph):
    if knn_graph is not None and knn_graph < A.shape[0]:
        idx = np.argsort(A, axis=0)[:-knn_graph, :]
        np.put_along_axis(A, idx, 0, axis=0)
        idx = np.argsort(A, axis=1)[:, :-knn_graph]
        np.put_along_axis(A, idx, 0, axis=1)
    return A

def spatial_graph(coord, img_size, knn_graph=32):
    coord = coord / np.array(img_size, np.float)
    dist = cdist(coord, coord)
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma**2)
    A[np.diag_indices_from(A)] = 0  # remove self-loops
    sparsify_graph(A, knn_graph)
    return A  # adjacency matrix (edges)

def visualize_superpixels(avg_values, superpixels):
    n_ch = avg_values.shape[1]
    img_sp = np.zeros((*superpixels.shape, n_ch))
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            img_sp[:, :, c][mask] = avg_values[sp, c]
    return img_sp


def superpixel_Alexnet_features(img, superpixels, alexnet):
    n_sp = len(np.unique(superpixels))
    img_alexnet = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    feature_alexnet = alexnet.features(img_alexnet)[0]
    n_ch = feature_alexnet.shape[0]

    feature_alexnet_resized = transform.resize(feature_alexnet.data.numpy(), (n_ch, img.shape[0], img.shape[1]))
    avg_values = np.zeros((n_sp, n_ch))
    coord = np.zeros((n_sp, 2))
    masks = []
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            avg_values[sp, c] = np.mean(feature_alexnet_resized[c, :, :][mask])
        coord[sp] = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        masks.append(mask)
    return avg_values, coord, masks

if __name__=='__main__':
    import data_loading
    from matplotlib import pyplot as plt
    from skimage.segmentation import slic
    import torchvision

    knn_graph = 32
    alexnet = torchvision.models.alexnet(pretrained=False)
    alexnet.load_state_dict(torch.load('./alexnet-owt-7be5be79.pth'))

    imgset = data_loading.load_imgs()
    img = imgset[51]
    superpixels = slic(img, n_segments=1000)
    avg_values, coord, masks = superpixel_Alexnet_features(img, superpixels, alexnet)
    print(avg_values.shape)
    A_spatial = spatial_graph(coord, img.shape[:2], knn_graph=knn_graph)  # keep only 16 neighbors for each node
    img_sp = visualize_superpixels(avg_values, superpixels)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.imshow(img_sp[:, :, 200:203])
    plt.title('$N=${} superpixels, mean alexnet features {} and coord {} features'.format(len(np.unique(superpixels)),
                                                                                          avg_values.shape,
                                                                                          coord.shape), fontsize=10)
    plt.subplot(122)
    plt.imshow(A_spatial ** 0.2)
    plt.colorbar()
    plt.title('Adjacency matrix of spatial edges')
    plt.show()
