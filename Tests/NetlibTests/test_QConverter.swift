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
        let converter = QConverter<UInt8, Float>(bias: 0, scale: UInt8.normScalef)
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: 127) == 0.5)
        XCTAssert(converter.convert(viewed: 0.5) == 127)
        XCTAssert(converter.convert(viewed: 1.0) == 255)
        XCTAssert(converter.convert(stored: 255) == 1.0)
    }

    //==========================================================================
    // test_convertUInt16Float
    func test_convertUInt16Float() {
        let converter = QConverter<UInt16, Float>(bias: 0, scale: UInt16.normScalef)
        XCTAssert(converter.convert(viewed: 0) == 0)
        XCTAssert(converter.convert(stored: 0) == 0)
        XCTAssert(converter.convert(stored: 32767) == 0.5)
        XCTAssert(converter.convert(viewed: 0.5) == 32767)
        XCTAssert(converter.convert(viewed: 1.0) == 65535)
        XCTAssert(converter.convert(stored: 65535) == 1.0)
    }
}
