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
//    static var allTests = [
//        ("test_convertUInt8Float", test_convertUInt8Float),
//    ]
//
//    //==========================================================================
//    // test_convertUInt8Float
//    func test_convertUInt8Float() {
//        let matrix = QMatrix<UInt8, Float>()
//        XCTAssert(matrix.convert(viewed: 0) == 0)
//        XCTAssert(matrix.convert(stored: 0) == 0)
//        XCTAssert(matrix.convert(stored: UInt8.max / 2) == 0.5)
//        XCTAssert(matrix.convert(viewed: 0.5) == UInt8.max / 2)
//        XCTAssert(matrix.convert(viewed: 1.0) == UInt8.max)
//        XCTAssert(matrix.convert(stored: UInt8.max) == 1.0)
//    }
//
//    //==========================================================================
//    // test_convertInt8Float
//    func test_convertInt8Float() {
//        let matrix = QMatrix<Int8, Float>()
//        XCTAssert(matrix.convert(viewed: 0) == 0)
//        XCTAssert(matrix.convert(stored: 0) == 0)
//        XCTAssert(matrix.convert(stored: Int8.max / 2) == 0.5)
//        XCTAssert(matrix.convert(viewed: 0.5) == Int8.max / 2)
//        XCTAssert(matrix.convert(viewed: 1.0) == Int8.max)
//        XCTAssert(matrix.convert(stored: Int8.max) == 1.0)
//        XCTAssert(matrix.convert(viewed: -1.0) == Int8.min)
//        XCTAssert(matrix.convert(stored: Int8.min) == -1.0)
//    }
//    
//    //==========================================================================
//    // test_convertInt8FloatBias
//    func test_convertInt8FloatBias() {
//        let bias: Float = 0.5
//        var matrix = QMatrix<Int8, Float>()
//        matrix.bias = bias
//        XCTAssert(matrix.convert(stored: 0) == bias)
//        XCTAssert(matrix.convert(viewed: bias) == 0)
//        XCTAssert(matrix.convert(stored: Int8.max / 2) == 0.5 + bias)
//        XCTAssert(matrix.convert(viewed: 0.5 + bias) == Int8.max / 2)
//        XCTAssert(matrix.convert(viewed: 1.0 + bias) == Int8.max)
//        XCTAssert(matrix.convert(stored: Int8.max) == 1.0 + bias)
//        XCTAssert(matrix.convert(viewed: -1.0 + bias) == Int8.min)
//        XCTAssert(matrix.convert(stored: Int8.min) == -1.0 + bias)
//    }
//    
//    //==========================================================================
//    // test_convertUInt16Float
//    func test_convertUInt16Float() {
//        let matrix = QMatrix<UInt16, Float>()
//        XCTAssert(matrix.convert(viewed: 0) == 0)
//        XCTAssert(matrix.convert(stored: 0) == 0)
//        XCTAssert(matrix.convert(stored: UInt16.max / 2) == 0.5)
//        XCTAssert(matrix.convert(viewed: 0.5) == UInt16.max / 2)
//        XCTAssert(matrix.convert(viewed: 1.0) == UInt16.max)
//        XCTAssert(matrix.convert(stored: UInt16.max) == 1.0)
//    }
//
////    //==========================================================================
////    // test_PixelQuantizing8
////    func test_PixelQuantizing8() {
////        let qv = UInt8.max / 4
////        let hv = UInt8.max / 2
////        let fv = UInt8.max
////        // pixel quantizer
////        let quantizer = QMatrix<RGBASample<UInt8>, RGBASample<Float>>()
////        let stored = RGBASample<UInt8>(r: 0, g: qv, b: hv, a: fv)
////        let viewed = RGBASample<Float>(r: 0, g: 0.25, b: 0.5, a: 1)
////        XCTAssert(quantizer.convert(viewed: viewed) == stored)
////        XCTAssert(quantizer.convert(stored: stored) == viewed)
////    }
//
////    //==========================================================================
////    // test_PixelQuantizing16
////    func test_PixelQuantizing16() {
////        let qv = UInt16.max / 4
////        let hv = UInt16.max / 2
////        let fv = UInt16.max
////        // pixel quantizer
////        let quantizer = QMatrix<RGBASample<UInt16>, RGBASample<Float>>()
////        let stored = RGBASample<UInt16>(r: 0, g: qv, b: hv, a: fv)
////        let viewed = RGBASample<Float>(r: 0, g: 0.25, b: 0.5, a: 1)
////
////        XCTAssert(quantizer.convert(viewed: viewed) == stored)
////        XCTAssert(quantizer.convert(stored: stored) == viewed)
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
////            let stored = Element(r: 0, g: qv, b: hv, a: fv)
////            let matrix = QMatrix<Element, Element>((2, 3),
////                                                 repeating: QMatrix(stored))
////
//////            let viewed = Element(r: 0, g: 0.25, b: 0.5, a: 1)
//////            let value = matrix.quantizer.convert(stored: stored)
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
