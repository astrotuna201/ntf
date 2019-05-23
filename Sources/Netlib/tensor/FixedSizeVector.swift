//******************************************************************************
//  Created by Edward Connell on 3/30/19
//  Copyright © 2019 Edward Connell. All rights reserved.
//
import Foundation

//==============================================================================
/// conformance indicates that scalar components are of the same type and
/// densely packed. This is necessary for zero copy view type casting of
/// short vector Element types.
/// For example: Matrix<RGBA<Float>> -> NHWCTensor<Float>
///
public protocol FixedSizeVector: Equatable {
    associatedtype Scalar
    static var count: Int { get }
}

public extension FixedSizeVector {
    static var count: Int {
        return MemoryLayout<Self>.size / MemoryLayout<Scalar>.size
    }
}

//==============================================================================
// RGB
public protocol RGBProtocol: FixedSizeVector {}

public struct RGB<Scalar>: RGBProtocol where Scalar: Numeric {
    public var r, g, b: Scalar

    @inlinable @inline(__always)
    public init() { r = Scalar.zero; g = Scalar.zero; b = Scalar.zero }

    @inlinable @inline(__always)
    public init(r: Scalar, g: Scalar, b: Scalar) {
        self.r = r; self.g = g; self.b = b
    }
}

//==============================================================================
// RGBA
public protocol RGBAProtocol: FixedSizeVector {}

public struct RGBA<Scalar> : RGBAProtocol where Scalar: Numeric {
    public var r, g, b, a: Scalar

    @inlinable @inline(__always)
    public init() {
        r = Scalar.zero
        g = Scalar.zero
        b = Scalar.zero
        a = Scalar.zero
    }
    
    @inlinable @inline(__always)
    public init(r: Scalar, g: Scalar, b: Scalar, a: Scalar) {
        self.r = r; self.g = g; self.b = b; self.a = a
    }
}

//==============================================================================
// Stereo
public protocol StereoProtocol: FixedSizeVector {}

public struct Stereo<Scalar>: StereoProtocol where Scalar: Numeric {
    public var left, right: Scalar

    @inlinable @inline(__always)
    public init() { left = Scalar.zero; right = Scalar.zero }

    @inlinable @inline(__always)
    public init(left: Scalar, right: Scalar) {
        self.left = left; self.right = right
    }
}
