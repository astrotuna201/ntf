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
        let converter = Quantizer<UInt8, Float>(bias: 0, scale: UInt8.normScalef)
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
        let converter = Quantizer<Int8, Float>(bias: 0, scale: Int8.normScalef)
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
    // test_convertInt8Float
    func test_convertInt8FloatBias() {
        let converter = Quantizer<Int8, Float>(bias: 0.5, scale: Int8.normScalef)
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
//        XCTAssert(converter.convert(stored: Int8.max / 2) == 0.5)
//        XCTAssert(converter.convert(viewed: 0.5) == Int8.max / 2)
//        XCTAssert(converter.convert(viewed: 1.0) == Int8.max)
//        XCTAssert(converter.convert(stored: Int8.max) == 1.0)
//        XCTAssert(converter.convert(viewed: -1.0) == Int8.min)
//        XCTAssert(converter.convert(stored: Int8.min) == -1.0)
    }
    
    //==========================================================================
    // test_convertUInt16Float
    func test_convertUInt16Float() {
        let converter = Quantizer<UInt16, Float>(bias: 0, scale: UInt16.normScalef)
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
        let converter = Quantizer<Float, UInt8>(bias: 0, scale: UInt8.normScalef)
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
        let converter = Quantizer<Float, Int8>(bias: 0, scale: Int8.normScalef)
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
        let converter = Quantizer<Float, UInt16>(bias: 0, scale: UInt16.normScalef)
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: 0.5) == UInt16.max / 2)
        XCTAssert(converter.convert(viewed: UInt16.max / 2) == 0.5)
        XCTAssert(converter.convert(viewed: UInt16.max) == 1.0)
        XCTAssert(converter.convert(stored: 1.0) == UInt16.max)
    }
}
