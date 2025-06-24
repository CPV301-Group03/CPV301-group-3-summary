# hog_descriptor_visualization.py

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure, color

# scikit-image có sẵn ảnh phi hành gia, rất tiện lợi
# Nếu muốn dùng URL, có thể dùng hàm load_image_from_url từ file trước.
print("Đang tải ảnh và tính toán HOG...")
image_color = data.astronaut()
image_gray = color.rgb2gray(image_color)

# 1. Tính toán HOG descriptor
# visualize=True để nhận lại ảnh HOG để hiển thị.
fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

# 2. Cải thiện độ tương phản của ảnh HOG để nhìn rõ hơn
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# 3. Hiển thị kết quả
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
plt.suptitle("Histogram of Oriented Gradients (HOG) Descriptor", fontsize=16)

ax1.axis('off')
ax1.imshow(image_color, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG Visualization')

plt.tight_layout()
plt.show()