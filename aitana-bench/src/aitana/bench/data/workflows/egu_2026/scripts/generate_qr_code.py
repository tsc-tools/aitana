import qrcode

# Email QR code content
email_content = "mailto:yannik.behr@earthsciences.nz?subject=EGU presentation 2026"

# Generate QR code
qr = qrcode.QRCode()
qr.add_data(email_content)
qr.make(fit=True)

# Save QR code as an image
img = qr.make_image(fill="black", back_color="white")
img.save(snakemake.output[0])
