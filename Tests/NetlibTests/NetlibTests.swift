import XCTest
import Foundation
import TensorFlow
@testable import Netlib
@testable import DeepLearning

final class NetlibTests: XCTestCase {
    static var allTests = [
        ("test_MnistTraining", test_MnistTraining),
        ("test_MnistInference", test_MnistInference)
    ]

    struct TrainingData {
        let trainingImages: TensorView<Float>
        let trainingLabels: TensorView<Int32>
        let testImages: TensorView<Float>
        let testLabels: TensorView<Int32>
    }

    public struct MNISTClassifier<ContextT>: Function where ContextT: Evaluating {
        @noDerivative public let functionId =
            UUID(uuidString: "7F2FFF4E-58D7-4ED9-A5D7-1E15D6D3D34A")!
        public var output: TensorView<Float>
        
        public init(context: ContextT,
                    inputShape: Netlib.TensorShape,
                    using stream: DeviceStream? = nil) {
        }
        
        @differentiable
        public func applied(to input: TensorView<Float>) -> TensorView<Float> {
            return input
        }

//        let maxPool1: MaxPool2D<Float>
//        let maxPool2: MaxPool2D<Float>
//        var conv1: Conv2D<Float>
//        var conv2: Conv2D<Float>
//        var dense1: Dense<Float>
//        var dense2: Dense<Float>
//
//        public init() {
//            conv1 = Conv2D(filterShape: (5, 5, 1, 20), padding: .valid)
//            maxPool1 = MaxPool2D(poolSize: (2, 2), strides: (2, 2), padding: .valid)
//            conv2 = Conv2D(filterShape: (5, 5, 20, 50), padding: .valid)
//            maxPool2 = MaxPool2D(poolSize: (2, 2), strides: (2, 2), padding: .valid)
//            dense1 = Dense(inputSize: 800, outputSize: 500, activation: relu)
//            dense2 = Dense(inputSize: 500, outputSize: 10, activation: { $0 })
//        }
//
//        // separate tensors
//        @differentiable(wrt: (self, input))
//        public func applied(to input: Tensor<Float>, in context: Context) -> Tensor<Float> {
//            let h0 = conv1.applied(to: input, in: context)
//            let h1 = maxPool1.applied(to: h0, in: context)
//            let h2 = conv2.applied(to: h1, in: context)
//            let h3 = maxPool2.applied(to: h2, in: context)
//            let dense1InputShape = Tensor<Int32>([h3.shape[0], 800])
//            let h4 = dense1.applied(to: h3.reshaped(toShape: dense1InputShape), in: context)
//            return dense2.applied(to: h4, in: context)
//        }
//
        public func infer(from input: TensorView<Float>) -> TensorView<Float> {
            return input
//            return softmax(applied(to: input))
        }
    }

    //==========================================================================
    //
    func test_MnistTraining() {
        let dir = "/home/ed/⁨nl2/⁨data/⁨mnist⁩/"

        //------------------------------------------------------------------------------
        // Load images and labels (without performing header validation)
        func readMNIST() -> TrainingData? {
            // validate
            func open(path: String) -> FileHandle? {
                return FileHandle(forReadingAtPath: dir + path)
            }

            guard let trainImagesFile = open(path: "train-images-idx3-ubyte"),
                let trainLabelsFile = open(path: "train-labels-idx1-ubyte"),
                let testImagesFile = open(path: "t10k-images-idx3-ubyte"),
                let testLabelsFile = open(path: "t10k-labels-idx1-ubyte") else {
                    return nil
            }

            func makeTensors(imagesFile: FileHandle, labelsFile: FileHandle) ->
                (TensorView<Float>, TensorView<Int32>) {
                    let imagesArray = imagesFile.readDataToEndOfFile().dropFirst(16).map { Float($0) }
                    let labelsArray = labelsFile.readDataToEndOfFile().dropFirst(8).map { Int32($0) }
                    let shape = TensorShape(labelsArray.count, 28, 28, 1)
                    let images = TensorView<Float>(shape: shape, scalars: imagesArray) // / 255
                    let labels = TensorView<Int32>(scalars: labelsArray)
                    return (images, labels)
            }

            let (trainImages, trainLabels) = makeTensors(imagesFile: trainImagesFile,
                                                         labelsFile: trainLabelsFile)
            let (testImages, testLabels) = makeTensors(imagesFile: testImagesFile,
                                                       labelsFile: testLabelsFile)

            return TrainingData(trainingImages: trainImages,
                                trainingLabels: trainLabels,
                                testImages: testImages,
                                testLabels: testLabels)
        }

        //------------------------------------------------------------------------------
        // load the data
        guard let data = readMNIST() else {
            print("Failed to open data files in working directory")
            exit(1)
        }

        //------------------------------------------------------------------------------
        // train classifier
        let batchSize = 60
        let batchShape = TensorShape(batchSize, 28, 28, 1)
        let trainingContext = TrainingContext()
        var model = MNISTClassifier(context: trainingContext, inputShape: batchShape)

        let optimizer = SGD<MNISTClassifier, Float>(learningRate: 0.01, momentum: 0.9)
        let trainingIterations = data.trainingImages.items / batchSize
        let epochs = 10

        let testBatchSize = 1000
        let testContext = InferenceContext()

        print("Begin training for \(epochs) epochs" )
        let start = Date()

        do {
            for epoch in 0..<epochs {
                //--------------------------------
                // train
                var totalLoss: Float = 0

                for i in 0..<trainingIterations {
                    let images = data.trainingImages.viewItems(offset: i, count: batchSize)
                    let labels = data.trainingLabels.viewItems(offset: i, count: batchSize)

                    let gradients = gradient(at: model) { model -> TensorView<Float> in
                        let logits = model.applied(to: images)
                        let batchLoss = TensorView<Float>() //softmaxCrossEntropy(logits: logits, labels: labels)
                        totalLoss += try batchLoss.scalarized()
                        return batchLoss
                    }
                    optimizer.update(&model.allDifferentiableVariables, along: gradients)
                }

                //--------------------------------
                // test
                var totalCorrect = 0
                for i in 0..<10 {
                    let images = data.testImages.viewItems(offset: i, count: testBatchSize)
                    let labels = data.testLabels.viewItems(offset: i, count: testBatchSize)
                    let predictions = model.infer(from: images)
    //                let correct = predictions.argmax(squeezingAxis: 1) .== labels
                    totalCorrect += 0 // Tensor<Int32>(correct).sum().scalarized()
                }

                let accuracy = Float(totalCorrect) / Float(data.testLabels.items)
                print("epoch \(epoch) accuracy: \(accuracy) loss: \(totalLoss)")
            }
            print("Training complete: \(String(timeInterval: Date().timeIntervalSince(start)))")
        } catch {
            
        }
    }

    func test_MnistInference() {
        //        XCTAssertEqual(Netlib().text, "Hello, World!")
    }
}
