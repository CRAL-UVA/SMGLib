# Importing this submodule causes the major Fusion classes
# such as Expression and Model to be extended to support
# arithmetic operations and indexing familiar from NumPy.
#
# For example, instead of
#
#   M.constraint("con", M.constraint(Expr.sub(x.index(i), Expr.mul(Z, b)), 
#                                    Domain.lessThan(0)))
#
# one can write
#
#  M.constraint("con", x[i] <= Z @ b)
#
# See the documentation and examples for more details.

from mosek.fusion import *
try:
    import numpy
    def isint(v):
        return isinstance(v,int) or isinstance(v,numpy.integer)
    def toindexarray(v):
        return numpy.array(v, dtype=numpy.int32)
except ImportError:
    def isint(v):
        return isinstance(v,int)
    def toindexarray(v):
        return v


#
# ARITHMETIC OPERATORS
#

# Required to take precedence over NumPy operator overloads
Expression.__array_ufunc__ = None  

# Addition
# e + o
Expression.__add__  = lambda self, other: Expr.add(self, other)
# o + e
Expression.__radd__ = lambda self, other: Expr.add(other, self)

# Subtraction
# e - o
Expression.__sub__  = lambda self, other: Expr.sub(self, other)
# o - e
Expression.__rsub__ = lambda self, other: Expr.sub(other, self)

# Unary operators
# -e
Expression.__neg__  = lambda self: Expr.neg(self)
# +e
Expression.__pos__  = lambda self: self

# Multiplication by scalar
# e * c
Expression.__mul__  = lambda self, other: Expr.mul(self, other)
# c * e
Expression.__rmul__ = lambda self, other: Expr.mul(other, self)

# Matrix/matrix, matrix/vector mutiplication
# e @ o
Expression.__matmul__  = lambda self, other: Expr.mul(self, other)
# o @ e
Expression.__rmatmul__ = lambda self, other: Expr.mul(other, self)

# Division, only by a constant
# e / c
Expression.__truediv__ = lambda self, other: Expr.mul(self, 1.0 / other)
# e / c
Expression.__div__     = lambda self, other: Expr.mul(self, 1.0 / other)

#
# CLAUSES
#

def exprdomopand(self,rhs):
    if isinstance(rhs,Term):
        return DJC.ANDFromTerms([self.toDJCTerm(),rhs.toDJCTerm()])
    elif isinstance(rhs,ExprDomain): 
        return DJC.ANDFromTerms([self.toDJCTerm(),rhs.toDJCTerm()])
    else:
        raise ValueError("Invalid argument for __and__")
    
ExprDomain.__and__ = exprdomopand
ExprDomain.__or__ = lambda self,other: DisjunctionTerms(self).__or__(other)
DisjunctionTerms.__or__ = lambda self,other: DisjunctionTerms(self,other) 
Term.__or__ = lambda self,other: DisjunctionTerms([self]).__or__(other)

#
# COMPARISON OPERATORS
#

# Equality: if the RHS is a domain then means membership:
#
#   expr == dom   <==>  expr \in dom
#
# Otherwise means equality
#
#  expr1 == expr2  <==>   expr1 - expr2 \in Domain.equalsTo(0)
def eq(self, other):
    if isinstance(other,LinearDomain):
        return ExprLinearDomain(self,other)
    elif isinstance(other,ConeDomain):
        return ExprConicDomain(self,other)
    elif isinstance(other,RangeDomain):
        return ExprRangeDomain(self,other)
    elif isinstance(other,PSDDomain):
        return ExprPSDDomain(self,other)
    elif isinstance(other,Expression):
        return ExprLinearDomain(Expr.sub(self,other), Domain.equalsTo(0.0))
    else:
        return ExprLinearDomain(self, Domain.equalsTo(other))

Expression.__eq__ = eq

# Inequalities

def le(self,other):
    if isinstance(other,Expression):
        return ExprLinearDomain(Expr.sub(self,other), Domain.lessThan(0.0))
    else:
        return ExprLinearDomain(self, Domain.lessThan(other))

def ge(self,other):
    if isinstance(other,Expression):
        return ExprLinearDomain(Expr.sub(self,other), Domain.greaterThan(0.0))
    else:
        return ExprLinearDomain(self, Domain.greaterThan(other))

Expression.__le__ = le
Expression.__ge__ = ge

#
# INDEXING AND SLICING
#

# These functions ensure proper handling of None and negative indices in slices sand picks/indexing
def convertslice(shape, key):
    start = [0 if key[i].start is None else shape[i]+key[i].start if key[i].start<0 else key[i].start for i in range(len(shape))]
    stop  = [shape[i] if key[i].stop is None else shape[i]+key[i].stop if key[i].stop<0 else key[i].stop for i in range(len(shape))]
    return (start, stop)

def convertindex(shape, key):
    if isint(key):
        return key if key>=0 else shape+key
    elif isinstance(key, list):
        return [key[i] if key[i]>=0 else shape[i]+key[i] for i in range(len(key))]
    else:
        raise ValueError("Invalid index argument")

def getitem(self, key):
    if isinstance(key, list):
        # Z[[i,j,k]] - sub-object given by an array of indices
        return self.pick([convertindex(self.shape[0], k) if isint(k) else convertindex(self.shape, list(k)) for k in key])
    elif isinstance(key, int):
        # Z[i] - single index
        return self.index(convertindex(self.shape[0], key))
    elif isinstance(key, tuple) and all(map(isint,key)):
        # Z[i, j, k] - single multidimensional index
        return self.index(convertindex(self.shape, list(key)))
    elif isinstance(key, slice):
        # Z[i:j] - slice
        return self.slice(*convertslice(self.shape, [key]))
    elif isinstance(key, tuple) and all(isint(k) or isinstance(k,slice) for k in key):
        # Z[i1:j1, i2, i3:j3] - multidimensional slices
        key = tuple( (slice(k,k+1) if isint(k) else k) for k in key)
        return self.slice(*convertslice(self.shape, key))
    else:
        # If nothing else then we asusme it is an array used for picking
        return self.pick(toindexarray(key))

# Slicing of expressions, variables and parameters
Expression.__getitem__ = getitem

# Slicing of constraints
Constraint.__getitem__ = getitem

#
# OTHER PROPERTIES
#

# Transposition
Variable.T   = property(fget = Variable.transpose)
Expression.T = property(fget = Expr.transpose)
Matrix.T     = property(fget = Matrix.transpose)

# Flattening
Variable.F   = property(fget = Var.flatten)
Expression.F = property(fget = Expr.flatten)

# Access to shape
Expression.shape = property(fget = Expression.getShape)
Variable.shape   = property(fget = Variable.getShape)


# Set many unnamed constraints or disjunctions at a time using.
# This is an undocumented helper function
#
#  Model.set(DisjunctiveClause[])
def modelset(self, clause):
    if isinstance(clause, ExprLinearDomain):
        return self.constraint(clause)
    elif isinstance(clause, ExprConicDomain):
        return self.constraint(clause)
    elif isinstance(clause, ExprRangeDomain):
        return self.constraint(clause)
    elif isinstance(clause, ExprPSDDomain):
        return self.constraint(clause)
    elif isinstance(clause, DisjunctionTerms):
        return self.disjunction(clause)
    else:
        raise f"Incorrect clause of type {type(clause)} passed to Model.set(). Expected: Clause or DisjunctiveClause."
Model.set = lambda self, *args: [modelset(self, clause) for clause in args]

