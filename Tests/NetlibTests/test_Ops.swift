//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
@testable import Netlib

class test_Ops: XCTestCase {
    static var allTests = [
        ("test_Casting", test_Casting),
        ("test_GetDefaultStream", test_GetDefaultStream),
        ("test_AddSubMulDiv", test_AddSubMulDiv),
    ]
    
    func test_Casting() {
//        do {
//            let a = VectorTensor<Float>(scalars: [1, 2, 3, 4])
//            let b = try VectorTensor<Int32>(a)
//            XCTAssert(a == b)
//
//        } catch {
//            XCTFail(String(describing: error))
//        }
    }
    
    func test_GetDefaultStream() {
        let stream = Platform.defaultStream
        XCTAssert(!stream.name.isEmpty)
    }
    
    func test_AddSubMulDiv() {
        let _ = MatrixTensor<RGBASample<UInt8>>(4, 3)
        let _ = VectorTensor<StereoSample<Int16>>(count: 1024)

        let m = MatrixTensor<UInt8>(4, 3)
        let a = VectorTensor<Float>(scalars: [1, 2, 3, 4])
        let b = VectorTensor<Float>(scalars: [4, 3, 2, 1])
        let y = VectorTensor<Float>(scalars: [0, 1, 2, 3])
        let expected = VectorTensor<Float>(scalars: [5, 5, 5, 5])

        do {
            //---------------------------------
            // unscoped, uses Platform.defaultStream
            let c1 = try a + b
            XCTAssert(c1 == expected)
            
            //---------------------------------
            // default stream scoped, uses Platform.defaultStream
            let c2 = try usingDefaultStream {
                return try a + b
            }
            XCTAssert(c2 == expected)

            //---------------------------------
            // other scoped
            // create a streams on each specified devices from the preferred service
            let stream = try Platform.global.createStreams(deviceIds: [0, 1])
            
            let c3 = try using(stream[0]) {
                return try a + b
            }
            XCTAssert(c3 == expected)

            //---------------------------------
            // sequential multi scoped
            let aPlusB = try using(stream[0]) {
                return try a + b
            }
            
            let aMinusB = try using(stream[1]) {
                return try a - b
            }
            
            let c4 = try usingDefaultStream {
                return try aPlusB + aMinusB
            }
            let c4expected = VectorTensor<Float>(scalars: [2, 4, 6, 8])
            // all three streams auto sync at this point
            XCTAssert(c4 == c4expected)

            //---------------------------------
            // syntax variations
            // transparent type conversion if the Scalar is compatible
            let _ = try a.pow(3)
            let _ = try a.pow(3.5)
            // integer array
            let _ = try m.pow(2) // okay
            // let _ = try m.pow(2.5)  Ints raised to float power won't compile
            let _ = try a + 1    // floats + broadcast Int is okay
            let _ = try m + 1
            // let _ = try m + 1.5  won't compile

            //---------------------------------
            // nested multi scoped
            let c5: VectorTensor<Float> = try usingDefaultStream {
                let x: VectorTensor<Float> = try using(stream[0]) {
                    let aMinusB = try using(stream[1]) {
                        return try a - b
                    }
                    // here stream[1] is synced with stream[0]
                    return try pow(a + b + aMinusB, y)
                }
                // can compose easily (but log(x) is computed twice in this case)
                let _ = try x.log() / (x.log() + 1)
                
                // temporary results are okay!
                let logx = try log(x)
                // here stream[0] is synced with the defaultStream
                return try logx / (logx + 1)
            }

            // all three streams auto sync at this point
            let c5expected = VectorTensor<Float>(scalars: [0, 0.581, 0.782, 0.862])
            let c5IsEqual = try c5.approximatelyEqual(to: c5expected).all().scalarValue()

            // here the defaultStream is synced with the app thread
            XCTAssert(c5IsEqual)

        } catch {
            XCTFail(String(describing: error))
        }
    }
}
