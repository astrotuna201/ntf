//******************************************************************************
//  Created by Edward Connell on 5/8/19
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_QConverter: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_UInt8Float", test_UInt8Float),
    ]

    //==========================================================================
    // test_UInt8FloatAdd
    func test_UInt8FloatAdd() {
        do {
            let matrix = QMatrix<UInt8, Float>((1, 4), any: 0..<4)
            let result = matrix + 1
            let values = try result.array()
            print(values)
//            XCTAssert(qvalues == qexpected)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_UInt8Float
    func test_UInt8Float() {
        do {
            let qv = UInt8.max / 4
            let hv = UInt8.max / 2
            let fv = UInt8.max
            let elements = [0, qv, hv, fv]
            
            let matrix = Matrix<UInt8>((1, 4), elements: elements)
            let values = try matrix.array()
            let expected: [UInt8] = [0, 63, 127, 255]
            XCTAssert(values == expected)
            
            let qmatrix = QMatrix<UInt8, Float>((1, 4), elements: elements)
            let qvalues = try qmatrix.array()
            let qexpected: [Float] = [0, 0.25, 0.5, 1.0]
            XCTAssert(qvalues == qexpected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_Int8Float
    func test_Int8Float() {
        do {
            let mv = Int8.min
            let zv = Int8.zero
            let qv = Int8.max / 4
            let hv = Int8.max / 2
            let fv = Int8.max
            let elements = [mv, zv, qv, hv, fv]
            
            let matrix = Matrix<Int8>((1, 5), elements: elements)
            let values = try matrix.array()
            let expected = [mv, zv, qv, hv, fv]
            XCTAssert(values == expected)
            
            let qmatrix = QMatrix<Int8, Float>((1, 5), elements: elements)
            let qvalues = try qmatrix.array()
            let qexpected: [Float] = [-1.0, 0, 0.25, 0.5, 1.0]
            XCTAssert(qvalues == qexpected)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_Int8FloatWithBias
    func test_Int8FloatWithBias() {
        let bias: Float = 0.5
        let mv = Int8.min
        let zv = Int8.zero
        let qv = Int8.max / 4
        let hv = Int8.max / 2
        let fv = Int8.max
        let elements = [mv, zv, qv, hv, fv]
        
        do {
            var matrix = QMatrix<Int8, Float>((1, 5), elements: elements)
            matrix.bias = bias
            let values = try matrix.array()
            let expected: [Float] = [-0.5, 0.5, 0.75, 1.0, 1.5]
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        do {
            let values: [Float] = [-0.5, 0.5, 0.75, 1.0, 1.5]
            let matrix = QMatrix<Int8, Float>((1, 5), scale: 1,
                                              bias: bias, values: values)
            let array = try matrix.elementArray()
            XCTAssert(array == elements)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
//    //==========================================================================
//    // test_convertUInt16Float
//    func test_convertUInt16Float() {
//        let matrix = QMatrix<UInt16, Float>()
//        XCTAssert(matrix.convert(viewed: 0) == 0)
//        XCTAssert(matrix.convert(element: 0) == 0)
//        XCTAssert(matrix.convert(element: UInt16.max / 2) == 0.5)
//        XCTAssert(matrix.convert(viewed: 0.5) == UInt16.max / 2)
//        XCTAssert(matrix.convert(viewed: 1.0) == UInt16.max)
//        XCTAssert(matrix.convert(element: UInt16.max) == 1.0)
//    }
//
//    //==========================================================================
//    // test_RGBAUInt8_RGBAFloat
//    func test_RGBUInt8_RGBFloat() {
//        typealias Image = QMatrix<RGB<UInt8>, RGB<Float>>
//        let zv = UInt8.zero
//        let hv = UInt8.max / 2
//        let fv = UInt8.max
//        let elements = [RGB<UInt8>(r: zv, g: hv, b: fv)]
//
//        do {
//            var image = Image((1, 1), elements: elements)
//            let values = try image.array()
//            let expected: [Float] = [0, 0.5, 1.0]
//            XCTAssert(values == expected)
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }

////    //==========================================================================
////    // test_PixelQuantizing16
////    func test_PixelQuantizing16() {
////        let qv = UInt16.max / 4
////        let hv = UInt16.max / 2
////        let fv = UInt16.max
////        // pixel quantizer
////        let quantizer = QMatrix<RGBASample<UInt16>, RGBASample<Float>>()
////        let element = RGBASample<UInt16>(r: 0, g: qv, b: hv, a: fv)
////        let viewed = RGBASample<Float>(r: 0, g: 0.25, b: 0.5, a: 1)
////
////        XCTAssert(quantizer.convert(viewed: viewed) == element)
////        XCTAssert(quantizer.convert(element: element) == viewed)
////    }
//
//    //==========================================================================
//    // test_matrixUInt8Float
//    func test_matrixUInt8Float() {
//        do {
//            let values: [UInt8] = [0, 63, 127, 255]
//            let matrix = QMatrix<UInt8, Float>((2, 2), values: values)
//            let values = try [Float](matrix.values())
//            //            let values = try matrix.array()
//            print(values)
//            let expected: [Float] = [0, 0.25, 0.5, 1]
//            XCTAssert(values == expected)
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
////    //==========================================================================
////    // test_matrixRGBA8
////    func test_matrixRGBA8() {
////        do {
////            typealias Element = RGBASample<UInt8>
////            typealias Element = RGBASample<Float>
////            let qv = UInt8.max / 4
////            let hv = UInt8.max / 2
////            let fv = UInt8.max
////            let element = Element(r: 0, g: qv, b: hv, a: fv)
////            let matrix = QMatrix<Element, Element>((2, 3),
////                                                 repeating: QMatrix(element))
////
//////            let viewed = Element(r: 0, g: 0.25, b: 0.5, a: 1)
//////            let value = matrix.quantizer.convert(element: element)
//////            XCTAssert(value == viewed)
////
////            let values = try [Element](matrix.values())
//////            let values = try matrix.array()
////            print(values)
////        } catch {
////            XCTFail(String(describing: error))
////        }
////    }
////

}
