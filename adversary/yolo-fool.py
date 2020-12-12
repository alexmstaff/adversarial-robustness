import tensorflow as tf
import foolbox as fb
import numpy as np
import eagerpy as ep
import matplotlib.pyplot as plt

# model = tf.keras.models.load_model('model.h5')
preprocessing = dict()
# bounds = (0, 1)

model = tf.keras.applications.ResNet50V2(weights="imagenet")
bounds = (-1, 1)

fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)

fmodel = fmodel.transform_bounds((0, 1))

images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=4)

print(fb.utils.accuracy(fmodel, images, labels))

images = ep.astensor(images)
labels = ep.astensor(labels)
# epsilons = np.linspace(0.0, 0.2, num=100)
epsilons = 0.006
attack = fb.attacks.LinfDeepFoolAttack()
# attack = fb.attacks.L2ProjectedGradientDescentAttack()
# attack = fb.attacks.L2CarliniWagnerAttack()

raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)


robust_accuracy = 1 - is_adv.float32().mean(axis=-1)

plt.plot(epsilons, robust_accuracy.numpy())

# print_img = clipped.raw
#
# print(tf.image.encode_png(print_img))
#
# for image in print_img:
#     # enc = tf.image.encode_png(image)
#     # print(enc)
#     break

fb.plot.images(images)
fb.plot.images(clipped)

print(fb.distances.l2(images, clipped))

plt.xlabel('epsilon')
plt.ylabel('adversarial accuracy')
# plt.axis([0, 0.01, 0, 1])
plt.show()
