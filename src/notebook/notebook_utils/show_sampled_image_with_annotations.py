X_test = []
for image in os.listdir(not_annotated_images_path):
  img = cv2.imread(os.path.join(not_annotated_images_path, str(image)))
  X_test.append(get_laplacian(img))
X_test = np.array(X_test)
print(X_test.shape)

y_cnn = model.predict(X_test)

plt.figure()
for i in range(len(X_test)):
    plt.axis('off')

    ny = y_cnn[i]

    image = cv2.rectangle(X_test[i],(int(ny[0]),int(ny[1])),(int(ny[2]),int(ny[3])),(255, 255, 255))
    print(os.listdir(not_annotated_images_path)[i])
    plt.imshow(image)
    plt.show()s