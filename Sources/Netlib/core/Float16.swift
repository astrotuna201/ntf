//******************************************************************************
//  Created by Edward Connell on 9/1/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import Foundation

public struct Float16: Equatable, Comparable {
	// initializers
	public init() {	bits = UInt16(0) }
	public init(_ value: Float16) { bits = value.bits }
	public init(bitPattern: UInt16) { bits = bitPattern }
	public init?(_ string: String) {
        guard let value = Float(string) else { return nil }
		bits = floatToFloat16_rn(value).bits
	}

    public init(_ value: Int8)   { bits = floatToFloat16_rn(Float(value)).bits}
	public init(_ value: UInt8)  { bits = floatToFloat16_rn(Float(value)).bits}
	public init(_ value: UInt16) { bits = floatToFloat16_rn(Float(value)).bits}
	public init(_ value: Int16)  { bits = floatToFloat16_rn(Float(value)).bits}
    public init(_ value: UInt32) { bits = floatToFloat16_rn(Float(value)).bits}
	public init(_ value: Int32)  { bits = floatToFloat16_rn(Float(value)).bits}
	public init(_ value: Int)    { bits = floatToFloat16_rn(Float(value)).bits}
	public init(_ value: Float)  { bits = floatToFloat16_rn(value).bits}
	public init(_ value: Double) { bits = floatToFloat16_rn(Float(value)).bits}
	public init(double: Double)  { bits = floatToFloat16_rn(Float(double)).bits}

	// properties
	var bits: UInt16

	// 10:5:1
	private static let mantissaMask: UInt16 = 0b0000001111111111
	private static let exponentMask: UInt16 = 0b0111110000000000
	private static let signMask: UInt16 = 0b1000000000000000

	// functions
	public var mantissa: Int { return (Int)(bits & Float16.mantissaMask) }
	public var exponent: Int { return (Int)(bits & Float16.exponentMask) }
	public var sign: Int { return (Int)(bits & Float16.signMask) }

    // operators
	public static func<(lhs: Float16, rhs: Float16) -> Bool {
		return Float(lhs) < Float(rhs)
	}

	public static func==(lhs: Float16, rhs: Float16) -> Bool {
		return lhs.bits == rhs.bits
	}

	public static func+(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) + Float(rhs))
	}

	public static func-(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) - Float(rhs))
	}

	public static func*(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) * Float(rhs))
	}

	public static func/(lhs: Float16, rhs: Float16) -> Float16 {
		return Float16(Float(lhs) / Float(rhs))
	}
}

//==============================================================================
// helpers
public func habs(_ value: Float16) -> Float16 {
    return Float16(bitPattern: value.bits & UInt16(0x7fff))
}

public func hneg(_ value: Float16) -> Float16 {
    return Float16(bitPattern: value.bits ^ UInt16(0x8000))
}

public func ishnan(_ value: Float16) -> Bool {
	// When input is NaN, exponent is all 1s and mantissa is non-zero.
	return (value.bits & UInt16(0x7c00)) ==
        UInt16(0x7c00) && (value.bits & UInt16(0x03ff)) != 0
}

public func ishinf(_ value: Float16) -> Bool {
	// When input is +/- inf, exponent is all 1s and mantissa is zero.
	return (value.bits & UInt16(0x7c00)) ==
        UInt16(0x7c00) && (value.bits & UInt16(0x03ff)) == 0
}

public func ishequ(bits: Float16, value: Float16) -> Bool {
	return !ishnan(bits) && !ishnan(value) && bits.bits == value.bits
}

public func hone() -> Float16 { return Float16(bitPattern: UInt16(0x3c00)) }

//==============================================================================
// extensions
extension Float {
	public init(_ fp16: Float16) { self = float16ToFloat(fp16) }
}

extension Int8 {
    public init(_ fp16: Float16) { self = Int8(Float(fp16)) }
}

extension UInt8 {
	public init(_ fp16: Float16) { self = UInt8(Float(fp16)) }
}

extension UInt16 {
	public init(_ fp16: Float16) { self = UInt16(Float(fp16)) }
}

extension Int16 {
	public init(_ fp16: Float16) { self = Int16(Float(fp16)) }
}

extension UInt32 {
    public init(_ fp16: Float16) { self = UInt32(Float(fp16)) }
}

extension Int32 {
	public init(_ fp16: Float16) { self = Int32(Float(fp16)) }
}

extension Int {
	public init(_ fp16: Float16) { self = Int(Float(fp16)) }
}

extension Double {
	public init(_ fp16: Float16) { self = Double(Float(fp16)) }
}

//==============================================================================
// floatToFloat16_rn
//	cpu functions for converting between FP32 and FP16 formats
// inspired from Paulius Micikevicius (pauliusm@nvidia.com)

public func floatToFloat16_rn(_ value: Float) -> Float16 {
	var result = Float16()

	let bits = value.bitPattern
	let ubits: UInt32 = bits & 0x7fffffff
	var remainder, shift, lsb, lsbS1, lsbM1: UInt32
	var sign, exponent, mantissa: UInt32

	// Get rid of +NaN/-NaN case first.
	if ubits > 0x7f800000 {
		result.bits = UInt16(0x7fff)
		return result
	}

	sign = ((bits >> 16) & UInt32(0x8000))

	// Get rid of +Inf/-Inf, +0/-0.
	if ubits > 0x477fefff {
		result.bits = UInt16(sign | UInt32(0x7c00))
		return result
	}
	if ubits < 0x33000001 {
		result.bits = UInt16(sign | 0x0000)
		return result
	}

	exponent = ((ubits >> 23) & 0xff)
	mantissa = (ubits & 0x7fffff)

	if exponent > 0x70 {
		shift = 13
		exponent -= 0x70
	} else {
		shift = 0x7e - exponent
		exponent = 0
		mantissa |= 0x800000
	}
	lsb    = (1 << shift)
	lsbS1 = (lsb >> 1)
	lsbM1 = (lsb - 1)

	// Round to nearest even.
	remainder = (mantissa & lsbM1)
	mantissa >>= shift
	if remainder > lsbS1 || (remainder == lsbS1 && (mantissa & 0x1) != 0) {
		mantissa += 1
		if (mantissa & 0x3ff) == 0 {
			exponent += 1
			mantissa = 0
		}
	}

	result.bits = UInt16(sign | (exponent << 10) | mantissa)
	return result
}

//==============================================================================
// float16ToFloat
public func float16ToFloat(_ value: Float16) -> Float {
	var sign = UInt32((value.bits >> 15) & 1)
	var exponent = UInt32((value.bits >> 10) & 0x1f)
	var mantissa = UInt32(value.bits & 0x3ff) << 13

	if exponent == 0x1f {  /* NaN or Inf */
		if mantissa != 0 {
			sign = 0
			mantissa = UInt32(0x7fffff)
		} else {
			mantissa = 0
		}
		exponent = 0xff
	} else if exponent == 0 {  /* Denorm or Zero */
		if mantissa != 0 {
			var msb: UInt32
			exponent = 0x71
			repeat {
				msb = (mantissa & 0x400000)
				mantissa <<= 1  /* normalize */
				exponent -= 1
			} while msb == 0
			mantissa &= 0x7fffff  /* 1.mantissa is implicit */
		}
	} else {
		exponent += 0x70
	}
	return Float(bitPattern: UInt32((sign << 31) | (exponent << 23) | mantissa))
}
