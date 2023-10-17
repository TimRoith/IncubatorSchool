import qrcode
import PIL
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import RoundedModuleDrawer, GappedSquareModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask

qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
qr.add_data('https://timroith.github.io/2023/09/28/ImageReconstruction.html')

img_1 = qr.make_image(image_factory=StyledPilImage, module_drawer=GappedSquareModuleDrawer())

img_1.save('QR.png')