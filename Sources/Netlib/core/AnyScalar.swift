//******************************************************************************
//  Created by Edward Connell on 10/11/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow

//==============================================================================
/// DataType
/// Used primarily for serialization, C APIs, and Cuda kernels
public enum DataType: Int {
    // integers
    case real8U, real8I, real16U, real16I, real32U, real32I, real64U, real64I
    // floats
    case real16F, real32F, real64F
    // non numeric
    case bool
}

//==============================================================================
/// AnyScalar
public protocol AnyScalar {
    init()
}

//==============================================================================
/// AnyNumeric
/// AnyNumeric enables the use of constants, type conversion, and
/// normalization within generics
///
public protocol AnyNumeric: AnyScalar {
	// unchanged cast value
	init(any: AnyNumeric)
	init?(string: String)
	var asUInt8  : UInt8   { get }
	var asUInt16 : UInt16  { get }
	var asInt16  : Int16   { get }
	var asInt32  : Int32   { get }
	var asUInt   : UInt    { get }
	var asInt    : Int     { get }
	var asFloat16: Float16 { get }
	var asFloat  : Float   { get }
	var asDouble : Double  { get }
	var asCVarArg: CVarArg { get }
	var asBool   : Bool    { get }

	// values are normalized to the new type during a cast
	init(norm any: AnyNumeric)
	var normUInt8  : UInt8   { get }
	var normUInt16 : UInt16  { get }
	var normInt16  : Int16   { get }
	var normInt32  : Int32   { get }
	var normUInt   : UInt    { get }
	var normInt    : Int     { get }
	var normFloat16: Float16 { get }
	var normFloat  : Float   { get }
	var normDouble : Double  { get }
	var normBool   : Bool    { get }

	var isFiniteValue: Bool { get }
    static var isFiniteType: Bool { get }
    static var dataType: DataType { get }
}

public protocol AnyInteger: AnyNumeric {}
public protocol AnyFloatingPoint: AnyNumeric {}

public typealias AnyTensorFlowScalar = AnyNumeric & TensorFlowScalar
public typealias AnyTensorFlowNumeric = AnyNumeric & TensorFlowNumeric
public typealias AnyTensorFlowInteger = AnyNumeric & TensorFlowInteger
public typealias AnyTensorFlowFloatingPoint = AnyNumeric & TensorFlowFloatingPoint

//------------------------------------------------------------------------------
extension UInt8: AnyInteger {
	public init(any: AnyNumeric) { self = any.asUInt8 }
	public var asUInt8  : UInt8  { return self }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normUInt8 }
	public static var normScale: Double = 1.0 / (Double(UInt8.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt8.max) + 1)
	
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * UInt8.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt8.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt8.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType { return .real8U }

	public init?(string: String) {
        guard let value = UInt8(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt16 : AnyInteger {
	public init(any: AnyNumeric) { self = any.asUInt16 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return self }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normUInt16 }
	public static var normScale: Double = 1.0 / (Double(UInt16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt16.max) + 1)

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * UInt16.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt16.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt16.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType { return .real16U }

    public init?(string: String) {
        guard let value = UInt16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int16 : AnyInteger {
	public init(any: AnyNumeric) { self = any.asInt16 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return self }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normInt16 }
	public static var normScale: Double = 1.0 / (Double(Int16.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int16.max) + 1)
	
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * Int16.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int16.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int16.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType { return .real16I }

	public init?(string: String) {
        guard let value = Int16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int32 : AnyInteger {
	public init(any: AnyNumeric) { self = any.asInt32 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return self }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normInt32 }
	public static var normScale: Double = 1.0 / (Double(Int32.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int32.max) + 1)

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * Int32.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int32.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int32.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType { return .real32I }

	public init?(string: String) {
        guard let value = Int32(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Int : AnyInteger {
	public init(any: AnyNumeric) { self = any.asInt }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return self }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normInt }
	public static var normScale: Double = 1.0 / (Double(Int.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(Int.max) + 1)
	
	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * Int.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * Int.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * Int.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType = {
        let index: [DataType] = [.real8I, .real16I, .real32I, .real64I]
        return index[MemoryLayout<Int>.size - 1]
    }()

	public init?(string: String) {
        guard let value = Int(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension UInt : AnyInteger {
	public init(any: AnyNumeric) { self = any.asUInt }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return self }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(any: self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normUInt }
	public static var normScale: Double = 1.0 / (Double(UInt.max) + 1)
	public static var normScalef: Float = Float(1.0) / (Float(UInt.max) + 1)

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return self == 0 ? Float16(0) : Float16((Float(self) + 1) * UInt.normScalef) }
	public var normFloat  : Float  { return self == 0 ? 0 : (Float(self) + 1) * UInt.normScalef }
	public var normDouble : Double { return self == 0 ? 0 : (Double(self) + 1) * UInt.normScale }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType = {
        let index: [DataType] = [.real8U, .real16U, .real32U, .real64U]
        return index[MemoryLayout<Int>.size - 1]
    }()

	public init?(string: String) {
        guard let value = UInt(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Bool: AnyNumeric {
	public init(any: AnyNumeric) { self = any.asBool }
	public var asUInt8  : UInt8  { return self ? 1 : 0 }
	public var asUInt16 : UInt16 { return self ? 1 : 0 }
	public var asInt16  : Int16  { return self ? 1 : 0 }
	public var asInt32  : Int32  { return self ? 1 : 0 }
	public var asUInt   : UInt   { return self ? 1 : 0 }
	public var asInt    : Int    { return self ? 1 : 0 }
	public var asFloat16: Float16{ return Float16(Float(self ? 1 : 0)) }
	public var asFloat  : Float  { return self ? 1 : 0 }
	public var asDouble : Double { return self ? 1 : 0 }
	public var asCVarArg: CVarArg{ return self.asInt }
	public var asBool   : Bool   { return self }
	public var asString : String { return self ? "true" : "false" }

	public init(norm any: AnyNumeric) { self = any.normBool }
	public static var normScale: Double = 1
	public static var normScalef : Float = 1

	public var normUInt8  : UInt8  { return asUInt8 }
	public var normUInt16 : UInt16 { return asUInt16 }
	public var normInt16  : Int16  { return asInt16 }
	public var normInt32  : Int32  { return asInt32 }
	public var normUInt   : UInt   { return asUInt }
	public var normInt    : Int    { return asInt }
	public var normFloat16: Float16{ return Float16(any: Float(any: self) * Bool.normScalef) }
	public var normFloat  : Float  { return Float(any: self) * Bool.normScalef }
	public var normDouble : Double { return Double(any: self) * Bool.normScale}
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return true }
    public static var isFiniteType: Bool { return true }
    public static var dataType: DataType { return .bool }

	public init?(string: String) {
        guard let value = Bool(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Float16 : AnyFloatingPoint {
	public init(any: AnyNumeric) { self = any.asFloat16 }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(any: self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return self }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return asFloat }
	public var asBool   : Bool   { return Float(self) != 0 }

	public init(norm any: AnyNumeric) { self = any.normFloat16 }
	
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat16: Float16{ return asFloat16 }
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return Float(self).isFinite }
    public static var isFiniteType: Bool { return false }
    public static var dataType: DataType { return .real16F }

	public init?(string: String) {
        guard let value = Float16(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Float : AnyFloatingPoint {
	public init(any: AnyNumeric) { self = any.asFloat }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return self }
	public var asDouble : Double { return Double(self) }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normFloat }
	
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat16: Float16{ return asFloat16 }
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return self.isFinite }
    public static var isFiniteType: Bool { return false }
    public static var dataType: DataType { return .real32F }

	public init?(string: String) {
        guard let value = Float(string) else { return nil }
		self = value
	}
}

//------------------------------------------------------------------------------
extension Double : AnyFloatingPoint {
	public init(any: AnyNumeric) { self = any.asDouble }
	public var asUInt8  : UInt8  { return UInt8(self) }
	public var asUInt16 : UInt16 { return UInt16(self) }
	public var asInt16  : Int16  { return Int16(self) }
	public var asInt32  : Int32  { return Int32(self) }
	public var asUInt   : UInt   { return UInt(self) }
	public var asInt    : Int    { return Int(self) }
	public var asFloat16: Float16{ return Float16(self) }
	public var asFloat  : Float  { return Float(self) }
	public var asDouble : Double { return self }
	public var asCVarArg: CVarArg{ return self }
	public var asBool   : Bool   { return self != 0 }

	public init(norm any: AnyNumeric) { self = any.normDouble }
	
	public var normUInt8  : UInt8  { return UInt8(Float(self)  * Float(UInt8.max))}
	public var normUInt16 : UInt16 { return UInt16(Float(self) * Float(UInt16.max))}
	public var normInt16  : Int16  { return Int16(Float(self)  * Float(Int16.max))}
	public var normInt32  : Int32  { return Int32(Float(self)  * Float(Int32.max))}
	public var normUInt   : UInt   { return UInt(Double(self)  * Double(UInt.max))}
	public var normInt    : Int    { return Int(Double(self)   * Double(Int.max))}
	public var normFloat16: Float16{ return asFloat16 }
	public var normFloat  : Float  { return asFloat }
	public var normDouble : Double { return asDouble }
	public var normBool   : Bool   { return asBool }

	public var isFiniteValue: Bool { return self.isFinite }
    public static var isFiniteType: Bool { return false }
    public static var dataType: DataType { return .real64F }

	public init?(string: String) {
        guard let value = Double(string) else { return nil }
		self = value
	}
}












