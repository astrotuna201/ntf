//******************************************************************************
//  Created by Edward Connell on 5/21/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Quantize: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_squaredDifference", test_squaredDifference),
        ("test_standardDeviation", test_standardDeviation),
        ("test_UInt8Float", test_UInt8Float),
        ("test_FloatUInt8", test_FloatUInt8),
    ]

    //==========================================================================
    // test_squaredDifference
    //
    func test_squaredDifference() {
        do {
            let vector = Vector<Float>(any: -1...3)
            let vm = Vector<Float>(with: vector.extents, repeating: mean(vector))
            let sd = squaredDifference(vector, vm)
            // -1, 0, 1, 2, 3
            let expected: [Float] = [4, 1, 0, 1, 4]
            let result = try sd.array()
            XCTAssert(result == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_standardDeviation
    //
    func test_standardDeviation() {
        do {
            let vector = Vector<Float>(any: -1...3)
            let (mean, std) = standardDeviation(vector)
            let meanValue = try mean.scalarValue()
            let stdValue = try std.scalarValue()
            XCTAssert(meanValue == 1 && stdValue == Float(1.41421354),
                      "got: \(meanValue), \(stdValue) expected: 1, 1.41421354")
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_UInt8Float
    //
    func test_UInt8Float() {
//        do {
//        } catch {
//            XCTFail(String(describing: error))
//        }
    }

    //==========================================================================
    // test_FloatUInt8
    //
    func test_FloatUInt8() {
        //        do {
        //        } catch {
        //            XCTFail(String(describing: error))
        //        }
    }
}
