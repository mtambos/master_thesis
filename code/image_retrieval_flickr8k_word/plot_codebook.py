from matplotlib import pyplot as plt
from multimodal_som import *
from sklearn.manifold import TSNE
import seaborn as sns


model = MultimodalSOM()
model.load('som_planar_rectangular')
projection = TSNE().fit_transform(model.som_model.codebook.reshape(1024, 4396))
plt.scatter(*projection.T, s=10)
sns.despine()
plt.savefig('codebook_tsne.png')
