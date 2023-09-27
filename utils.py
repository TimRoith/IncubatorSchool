import skimage as ski

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))