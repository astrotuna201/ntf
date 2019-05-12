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
        ("test_convertUInt8Float", test_convertUInt8Float),
    ]

    //==========================================================================
    // test_convertUInt8Float
    func test_convertUInt8Float() {
        var converter = Quantizer<UInt8, Float>()
        converter.updateScales()
//        converter.convert(viewed: 0)
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: UInt8.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: 0.5) == UInt8.max / 2)
        XCTAssert(converter.convert(viewed: 1.0) == UInt8.max)
        XCTAssert(converter.convert(stored: UInt8.max) == 1.0)
    }

    //==========================================================================
    // test_convertInt8Float
    func test_convertInt8Float() {
        let converter = Quantizer<Int8, Float>()
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: Int8.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: 0.5) == Int8.max / 2)
        XCTAssert(converter.convert(viewed: 1.0) == Int8.max)
        XCTAssert(converter.convert(stored: Int8.max) == 1.0)
        XCTAssert(converter.convert(viewed: -1.0) == Int8.min)
        XCTAssert(converter.convert(stored: Int8.min) == -1.0)
    }
    
    //==========================================================================
    // test_convertInt8FloatBias
    func test_convertInt8FloatBias() {
        let bias: Float = 0.5
        let converter = Quantizer<Int8, Float>(bias: bias)
        XCTAssert(converter.convert(stored: 0) == bias)
        XCTAssert(converter.convert(viewed: bias) == 0)
        XCTAssert(converter.convert(stored: Int8.max / 2) == 0.5 + bias)
        XCTAssert(converter.convert(viewed: 0.5 + bias) == Int8.max / 2)
        XCTAssert(converter.convert(viewed: 1.0 + bias) == Int8.max)
        XCTAssert(converter.convert(stored: Int8.max) == 1.0 + bias)
        XCTAssert(converter.convert(viewed: -1.0 + bias) == Int8.min)
        XCTAssert(converter.convert(stored: Int8.min) == -1.0 + bias)
    }
    
    //==========================================================================
    // test_convertUInt16Float
    func test_convertUInt16Float() {
        let converter = Quantizer<UInt16, Float>()
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: UInt16.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: 0.5) == UInt16.max / 2)
        XCTAssert(converter.convert(viewed: 1.0) == UInt16.max)
        XCTAssert(converter.convert(stored: UInt16.max) == 1.0)
    }

    //==========================================================================
    // test_convertFloatUInt8
    func test_convertFloatUInt8() {
        let converter = Quantizer<Float, UInt8>()
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: 0.5) == UInt8.max / 2)
        XCTAssert(converter.convert(viewed: UInt8.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: UInt8.max) == 1.0)
        XCTAssert(converter.convert(stored: 1.0) == UInt8.max)
    }
    
    //==========================================================================
    // test_convertInt8Float
    func test_convertFloatInt8() {
        let converter = Quantizer<Float, Int8>()
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: 0.5) == Int8.max / 2)
        XCTAssert(converter.convert(viewed: Int8.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: Int8.max) == 1.0)
        XCTAssert(converter.convert(stored: 1.0) == Int8.max)
        XCTAssert(converter.convert(viewed: Int8.min) == -1.0)
        XCTAssert(converter.convert(stored: -1.0) == Int8.min)
    }
    
    //==========================================================================
    // test_convertUInt16Float
    func test_convertFloatUInt16() {
        let converter = Quantizer<Float, UInt16>()
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: 0.5) == UInt16.max / 2)
        XCTAssert(converter.convert(viewed: UInt16.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: UInt16.max) == 1.0)
        XCTAssert(converter.convert(stored: 1.0) == UInt16.max)
    }
    
    //==========================================================================
    // test_PixelQuantizing8
    func test_PixelQuantizing8() {
        let qv = UInt8.max / 4
        let hv = UInt8.max / 2
        let fv = UInt8.max
        // pixel quantizer
        let quantizer = Quantizer<RGBASample<UInt8>, RGBASample<Float>>()
        let stored = RGBASample<UInt8>(r: 0, g: qv, b: hv, a: fv)
        let viewed = RGBASample<Float>(r: 0, g: 0.25, b: 0.5, a: 1)
        
        XCTAssert(quantizer.convert(viewed: viewed) == stored)
        XCTAssert(quantizer.convert(stored: stored) == viewed)
    }

    //==========================================================================
    // test_PixelQuantizing16
    func test_PixelQuantizing16() {
        let qv = UInt16.max / 4
        let hv = UInt16.max / 2
        let fv = UInt16.max
        // pixel quantizer
        let quantizer = Quantizer<RGBASample<UInt16>, RGBASample<Float>>()
        let stored = RGBASample<UInt16>(r: 0, g: qv, b: hv, a: fv)
        let viewed = RGBASample<Float>(r: 0, g: 0.25, b: 0.5, a: 1)
        
        XCTAssert(quantizer.convert(viewed: viewed) == stored)
        XCTAssert(quantizer.convert(stored: stored) == viewed)
    }

    //==========================================================================
    // test_perfPixelQuantizing8
    func test_perfPixelQuantizing8() {
        do {
            typealias SourceMatrix = QMatrix<RGBASample<UInt8>,RGBASample<Float>>
            let qv = UInt8.max / 4
            let hv = UInt8.max / 2
            let fv = UInt8.max
            
            let stored = RGBASample<UInt8>(r: 0, g: qv, b: hv, a: fv)
            let viewed = RGBASample<Float>(r: 0, g: 0.25, b: 0.5, a: 1)
            
            let rows = 2
            let cols = 3
            let matrix = SourceMatrix((rows, cols), repeating: QMatrix(stored))
            
            let value = matrix.quantizer.convert(stored: stored)
            XCTAssert(value == viewed)

            
            
            let values = try [RGBASample<Float>](matrix.values())
//            let values = try matrix.array()
            print(values)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    

}
