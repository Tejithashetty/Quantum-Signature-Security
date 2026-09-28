from src.signature_security import SignatureSecurity


security = SignatureSecurity()

message = b"Quantum Signature Security"

signature = security.sign(message)

print("Original message:")
print(message.decode())

print("\nSignature generated successfully.")

print(
    "\nOriginal verification:",
    security.verify(message, signature)
)


tampered_message = b"Modified Quantum Signature Security"

print(
    "\nTampered verification:",
    security.verify(
        tampered_message,
        signature
    )
)