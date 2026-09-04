try:
    from PIL import Image, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageEnhance = None



class ChartPreprocessor:
    """
    Tiền xử lý ảnh biểu đồ:
    - Resize thông minh giữ tỉ lệ (Aspect Ratio).
    - Tăng độ tương phản để làm nổi bật râu nến và nhãn trục.
    """

    def __init__(self, max_size: int = 512, contrast_factor: float = 1.2):
        self.max_size = max_size
        self.contrast_factor = contrast_factor

    def process(self, image: Image.Image) -> Image.Image:
        # Giữ tỉ lệ, kích thước cạnh lớn nhất là max_size
        image.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)

        # Tăng tương phản nhẹ nhàng cho các nét mảnh (râu nến, đường chỉ báo MA)
        if self.contrast_factor != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.contrast_factor)

        return image
