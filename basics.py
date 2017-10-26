from sklearn.datasets import load_digits

digits = load_digits()
print (type(digits.data))
print (type(digits.target))
print (type(digits.target_names))

print (digits.data.shape)
print (digits.target.shape)

X = digits.data
y = digits.target

