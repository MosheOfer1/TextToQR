import qrcode
import random
import string


def generate_qr_code(data):
    qr = qrcode.QRCode(version=None, box_size=1, border=0)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.version


def random_string(length, uppercase_only=False):
    if uppercase_only:
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    else:
        return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))


def find_max_chars(uppercase_only=False):
    low, high = 1, 100
    while low <= high:
        mid = (low + high) // 2
        test_string = random_string(mid, uppercase_only)
        version = generate_qr_code(test_string)
        if version == 1:
            low = mid + 1
        else:
            high = mid - 1
    return high


# Test for uppercase only (Alphanumeric mode)
max_chars_uppercase = find_max_chars(uppercase_only=True)
print(f"Maximum characters (uppercase only) for 21x21 QR code: {max_chars_uppercase}")

# Test for mixed case (Byte mode)
max_chars_mixed = find_max_chars(uppercase_only=False)
print(f"Maximum characters (mixed case) for 21x21 QR code: {max_chars_mixed}")

# Verify the results
for mode, max_chars in [("Uppercase", max_chars_uppercase), ("Mixed case", max_chars_mixed)]:
    test_string = random_string(max_chars, mode == "Uppercase")
    version = generate_qr_code(test_string)
    print(f"\nVerification for {mode}:")
    print(f"String length: {len(test_string)}")
    print(f"QR Version: {version}")
