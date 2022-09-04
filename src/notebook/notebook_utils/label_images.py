X_test = []
for image in os.listdir(not_annotated_images_path):
  img = cv2.imread(os.path.join(not_annotated_images_path, str(image)))
  img = current_function(img)
  X_test.append(img)
X_test = np.array(X_test)
print(X_test.shape)

y_cnn = model.predict(X_test)

plt.figure()
for i in range(len(X_test)):
    plt.axis('off')

    ny = y_cnn[i]
    print(ny)

    image = cv2.rectangle(X_test[i],(int(ny[2]),int(ny[3])),(int(ny[0]),int(ny[1])),(255, 0, 0))
    print(os.listdir(not_annotated_images_path)[i])
    plt.imshow(image)
    plt.show()