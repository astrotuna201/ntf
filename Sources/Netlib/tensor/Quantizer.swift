//==============================================================================
/// QuantizeConverting
public protocol QuantizeConverting {
    associatedtype Stored
    associatedtype Viewed
    
    /// the bias to apply during conversion
    var bias: Float { get }
    /// the scale to apply during conversion
    var scale: Float { get }
    var oneOverScale: Float { get }
    /// converts from Scalar to ViewedScalar
    func convert(stored: Stored) -> Viewed
    /// converts from Scalar to ViewedScalar
    func convert(viewed: Viewed) -> Stored
}

//==============================================================================
/// Quantizer
public struct Quantizer<Stored, Viewed>: QuantizeConverting {
    public let bias: Float
    public let scale: Float
    public let oneOverScale: Float
    
    public init(scale: Float, bias: Float = 0) {
        self.bias = bias
        self.scale = scale
        self.oneOverScale = 1 / scale
    }
}

//==============================================================================
/// QuantizeConverting
public extension QuantizeConverting {
    func convert(stored: Stored) -> Viewed { fatalError("not implemented") }
    func convert(viewed: Viewed) -> Stored { fatalError("not implemented") }
    
    //==========================================================================
    /// NOTE: It's likely most of the time value will be Float, so the cast of
    /// TF(bias) should be thrown out by the compiler. In the case of
    /// something like Float16, it will likely be faster to cast anyway.
    @inlinable @inline(__always)
    func Float2Int<TF, TI>(value: TF) -> TI
        where TF: BinaryFloatingPoint, TI: BinaryInteger
    {
        if value == TF(bias) {
            return 0
        } else if value > 0 {
            return TI(((Float(value) - bias) * oneOverScale) - 1)
        } else {
            return TI((Float(value) - bias) * oneOverScale)
        }
    }

    //==========================================================================
    @inlinable @inline(__always)
    func Int2Float<TI, TF>(value: TI) -> TF
        where TI: BinaryInteger, TF: BinaryFloatingPoint
    {
        if value == 0 {
            return TF(bias)
        } else if value > 0 {
            return TF((Float(value) + 1) * scale + bias)
        } else {
            return TF((Float(value)) * scale + bias)
        }
    }
}

//==============================================================================
/// Integer --> Float
public extension QuantizeConverting where
    Stored: BinaryInteger, Viewed: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Int2Float(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Float2Int(value: viewed)
    }
}

//==============================================================================
/// Float --> Integer -->
public extension QuantizeConverting where
    Stored: BinaryFloatingPoint, Viewed: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Float2Int(value: stored)
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Int2Float(value: viewed)
    }
}

//==============================================================================
/// UniformDenseScalar2
public extension QuantizeConverting where
    Stored: UniformDenseScalar2, Stored.Component: BinaryInteger,
    Viewed: UniformDenseScalar2, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Int2Float(value: stored.c0),
                      c1: Int2Float(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Float2Int(value: viewed.c0),
                      c1: Float2Int(value: viewed.c1))
    }
}

public extension QuantizeConverting where
    Stored: UniformDenseScalar2, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar2, Viewed.Component: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Float2Int(value: stored.c0),
                      c1: Float2Int(value: stored.c1))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Int2Float(value: viewed.c0),
                      c1: Int2Float(value: viewed.c1))
    }
}
//==============================================================================
/// UniformDenseScalar3
public extension QuantizeConverting where
    Stored: UniformDenseScalar3, Stored.Component: BinaryInteger,
    Viewed: UniformDenseScalar3, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Int2Float(value: stored.c0),
                      c1: Int2Float(value: stored.c1),
                      c2: Int2Float(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Float2Int(value: viewed.c0),
                      c1: Float2Int(value: viewed.c1),
                      c2: Float2Int(value: viewed.c2))
    }
}

public extension QuantizeConverting where
    Stored: UniformDenseScalar3, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar3, Viewed.Component: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Float2Int(value: stored.c0),
                      c1: Float2Int(value: stored.c1),
                      c2: Float2Int(value: stored.c2))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Int2Float(value: viewed.c0),
                      c1: Int2Float(value: viewed.c1),
                      c2: Int2Float(value: viewed.c2))
    }
}

//==============================================================================
/// UniformDenseScalar4
public extension QuantizeConverting where
    Stored: UniformDenseScalar4, Stored.Component: BinaryInteger,
    Viewed: UniformDenseScalar4, Viewed.Component: BinaryFloatingPoint
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Int2Float(value: stored.c0),
                      c1: Int2Float(value: stored.c1),
                      c2: Int2Float(value: stored.c2),
                      c3: Int2Float(value: stored.c3))
    }

    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Float2Int(value: viewed.c0),
                      c1: Float2Int(value: viewed.c1),
                      c2: Float2Int(value: viewed.c2),
                      c3: Float2Int(value: viewed.c3))
    }
}

public extension QuantizeConverting where
    Stored: UniformDenseScalar4, Stored.Component: BinaryFloatingPoint,
    Viewed: UniformDenseScalar4, Viewed.Component: BinaryInteger
{
    @inlinable @inline(__always)
    func convert(stored: Stored) -> Viewed {
        return Viewed(c0: Float2Int(value: stored.c0),
                      c1: Float2Int(value: stored.c1),
                      c2: Float2Int(value: stored.c2),
                      c3: Float2Int(value: stored.c3))
    }
    
    @inlinable @inline(__always)
    func convert(viewed: Viewed) -> Stored {
        return Stored(c0: Int2Float(value: viewed.c0),
                      c1: Int2Float(value: viewed.c1),
                      c2: Int2Float(value: viewed.c2),
                      c3: Int2Float(value: viewed.c3))
    }
}

