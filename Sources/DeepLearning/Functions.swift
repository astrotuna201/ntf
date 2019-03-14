//******************************************************************************
//  Created by Edward Connell on  3/10/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation
import TensorFlow
import Netlib

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
@_fixed_layout
public struct Dense2<Scalar: TensorFlowFloatingPoint> {
    /// the unique identifier for this function
//    @noDerivative
    public let functionId = UUID(uuidString: "acf809d5-3fc0-41bb-879b-622f32356d91")!
    /// the output view for this instance
//    @noDerivative
    public var output: TensorView<Scalar>
    /// The weight matrix.
    public var weight: TensorView<Scalar>
    /// The bias vector.
    public var bias: TensorView<Scalar>
    public typealias Activation =
//            @differentiable
            (TensorView<Scalar>) -> TensorView<Scalar>
    /// The element-wise activation function.
//    @noDerivative
    public let activation: Activation

    public init(weight: TensorView<Scalar>,
                bias: TensorView<Scalar>,
                activation: @escaping Activation,
                using stream: DeviceStream) {
        // store parameters
        self.weight = weight
        self.bias = bias
        self.activation = activation
        // create output view TODO: give it a real shape
        output = TensorView<Scalar>()
    }

//    @differentiable
    public func applied(to input: TensorView<Scalar>) -> TensorView<Scalar> {
        return TensorView<Scalar>() // activation(matmul(input, weight) + bias)
    }
}

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
@_fixed_layout
public struct Dense<Scalar: TensorFlowFloatingPoint>: Function {
    /// the unique identifier for this function
    @noDerivative
    public let functionId = UUID(uuidString: "acf809d5-3fc0-41bb-879b-622f32356d91")!
    /// the output view for this instance
    @noDerivative
    public var output: Tensor<Scalar>
    /// The weight matrix.
    public var weight: Tensor<Scalar>
    /// The bias vector.
    public var bias: Tensor<Scalar>
    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    /// The element-wise activation function.
    @noDerivative
    public let activation: Activation

    public init(weight: Tensor<Scalar>,
                bias: Tensor<Scalar>,
                activation: @escaping Activation,
                using stream: DeviceStream) {
        // store parameters
        self.weight = weight
        self.bias = bias
        self.activation = activation
        // create output view TODO: give it a real shape
        output = Tensor<Scalar>()
    }

    @differentiable
    public func applied(to input: Tensor<Scalar>) -> Tensor<Scalar> {
        return activation(matmul(input, weight) + bias)
    }
}

//public extension Dense {
//    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
//    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
//    /// is initialized using Glorot uniform initialization with the specified generator. The bias
//    /// vector is created with shape `[outputSize]` and is initialized with zeros.
//    init<G: RandomNumberGenerator>(
//            inputSize: Int,
//            outputSize: Int,
//            activation: @escaping Activation = identity,
//            generator: inout G
//    ) {
//        self.init(weight: Tensor(glorotUniform: [Int32(inputSize), Int32(outputSize)],
//                generator: &generator),
//                bias: Tensor(zeros: [Int32(outputSize)]),
//                activation: activation)
//    }
//
//    init(inputSize: Int, outputSize: Int, activation: @escaping Activation = identity) {
//        self.init(inputSize: inputSize, outputSize: outputSize, activation: activation,
//                generator: &PhiloxRandomNumberGenerator.global)
//    }
//}
//
//public extension Dense {
//    /// Creates a `Dense` layer with the specified input size, output size, and element-wise
//    /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
//    /// is initialized using Glorot uniform initialization with the specified seed. The bias vector
//    /// is created with shape `[outputSize]` and is initialized with zeros.
//    init(
//            inputSize: Int,
//            outputSize: Int,
//            activation: @escaping Activation = identity,
//            seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
//                    Int64.random(in: Int64.min..<Int64.max))
//    ) {
//        self.init(weight: Tensor(glorotUniform: [Int32(inputSize), Int32(outputSize)],
//                seed: seed),
//                bias: Tensor(zeros: [Int32(outputSize)]),
//                activation: activation)
//    }
//}
//
///// A convolutional neural network layer.
//@_fixed_layout
//public struct Conv2D<Scalar: TensorFlowFloatingPoint>: Layer {
//    /// The 4-D convolution kernel.
//    public var filter: Tensor<Scalar>
//    /// The bias vector.
//    public var bias: Tensor<Scalar>
//    public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
//    /// The element-wise activation function.
//    @noDerivative public let activation: Activation
//    /// The strides of the sliding window for each dimension of a 4-D input.
//    /// Strides in non-spatial dimensions must be `1`.
//    @noDerivative public let strides: (Int32, Int32)
//    /// The padding algorithm for convolution.
//    @noDerivative public let padding: Padding
//
//    public init(
//            filter: Tensor<Scalar>,
//            bias: Tensor<Scalar>,
//            activation: @escaping Activation,
//            strides: (Int, Int),
//            padding: Padding
//    ) {
//        self.filter = filter
//        self.bias = bias
//        self.activation = activation
//        (self.strides.0, self.strides.1) = (Int32(strides.0), Int32(strides.1))
//        self.padding = padding
//    }
//
//    @differentiable
//    public func applied(to input: Tensor<Scalar>, in _: Context) -> Tensor<Scalar> {
//        return activation(input.convolved2D(withFilter: filter,
//                strides: (1, strides.0, strides.1, 1),
//                padding: padding) + bias)
//    }
//}
//
//public extension Conv2D {
//    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, and
//    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
//    /// initialization with the specified generator. The bias vector is initialized with zeros.
//    init<G: RandomNumberGenerator>(
//            filterShape: (Int, Int, Int, Int),
//            strides: (Int, Int) = (1, 1),
//            padding: Padding = .valid,
//            activation: @escaping Activation = identity,
//            generator: inout G
//    ) {
//        let filterTensorShape = TensorShape([
//            Int32(filterShape.0), Int32(filterShape.1),
//            Int32(filterShape.2), Int32(filterShape.3)])
//        self.init(
//                filter: Tensor(glorotUniform: filterTensorShape, generator: &generator),
//                bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
//                activation: activation,
//                strides: strides,
//                padding: padding)
//    }
//
//    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, and
//    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
//    /// initialization. The bias vector is initialized with zeros.
//    init(
//            filterShape: (Int, Int, Int, Int),
//            strides: (Int, Int) = (1, 1),
//            padding: Padding = .valid,
//            activation: @escaping Activation = identity
//    ) {
//        self.init(filterShape: filterShape, strides: strides, padding: padding,
//                activation: activation,
//                generator: &PhiloxRandomNumberGenerator.global)
//    }
//}
//
//public extension Conv2D {
//    /// Creates a `Conv2D` layer with the specified filter shape, strides, padding, and
//    /// element-wise activation function. The filter tensor is initialized using Glorot uniform
//    /// initialization with the specified seed. The bias vector is initialized with zeros.
//    init(
//            filterShape: (Int, Int, Int, Int),
//            strides: (Int, Int) = (1, 1),
//            padding: Padding = .valid,
//            activation: @escaping Activation = identity,
//            seed: (Int64, Int64) = (Int64.random(in: Int64.min..<Int64.max),
//                    Int64.random(in: Int64.min..<Int64.max))
//    ) {
//        let filterTensorShape = TensorShape([
//            Int32(filterShape.0), Int32(filterShape.1),
//            Int32(filterShape.2), Int32(filterShape.3)])
//        self.init(
//                filter: Tensor(glorotUniform: filterTensorShape, seed: seed),
//                bias: Tensor(zeros: TensorShape([Int32(filterShape.3)])),
//                activation: activation,
//                strides: (Int32(strides.0), Int32(strides.1)),
//                padding: padding)
//    }
//}
