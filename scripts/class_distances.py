import numpy as np
from scipy.stats import norm, wasserstein_distance, energy_distance
import matplotlib.pyplot as plt
from dictances import bhattacharyya

# Generate some data for this demonstration.

from trento_config import (
    dataset,
)

X,y = dataset.full_dataset 
print(X.shape, y.shape, np.unique(y))
class0 = X[y==0][:,60]
class1 = X[y==1][:,60]
class2 = X[y==2][:,60]
class3 = X[y==3][:,60]
class4 = X[y==4][:,60]
class5 = X[y==5][:,60]
class6 = X[y==6][:,60]

print(wasserstein_distance(class1,class0))
print(wasserstein_distance(class1,class1))
print(wasserstein_distance(class1,class2))
print(wasserstein_distance(class1,class3))
print(wasserstein_distance(class1,class4))
print(wasserstein_distance(class1,class5))
print(wasserstein_distance(class1,class6))


print(energy_distance(class1,class0))
print(energy_distance(class1,class1))
print(energy_distance(class1,class2))
print(energy_distance(class1,class3))
print(energy_distance(class1,class4))
print(energy_distance(class1,class5))
print(energy_distance(class1,class6))

dict_class1 = {'a':class1}
dict_class0 = {'a':class0}

print(bhattacharyya(dict_class1,dict_class0))


# # Fit a normal distribution to the data:
# mu, std = norm.fit(data)

# # Plot the histogram.
# plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# # Plot the PDF.
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.show()
