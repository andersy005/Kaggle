class Set:

	def __init__(self, values=None):
		self.dict = {}

		if values is not None:
			for value in values:
				self.add(value)


	def __repr__(self):
		return "Set: " + str(self.dict.keys())

	def add(self, value):
		self.dict[value] = True

	def contains(self, value):
		return value in self.dict

	def remove(self, value):
		del self.dict[value]



a = Set([1,2,3])
a.add(4)

print a
print a.contains(0)
