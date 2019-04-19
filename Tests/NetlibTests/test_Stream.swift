//******************************************************************************
//  Created by Edward Connell on 3/23/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import XCTest
import Foundation
@testable import Netlib

class test_Stream: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_GetDefaultStream", test_GetDefaultStream),
        ("test_AddSubMulDiv", test_AddSubMulDiv),
    ]
    
    //==========================================================================
    // test_GetDefaultStream
    func test_GetDefaultStream() {
        let stream = Platform.defaultStream
        XCTAssert(!stream.name.isEmpty)
    }
    
    //==========================================================================
    // test_AddSubMulDiv
    func test_AddSubMulDiv() {
        // handle unexpected stream error
        handleStreamExceptions { error in
            XCTFail(String(describing: error))
        }
        
        // create some tensors
        let _ = Matrix<RGBASample<UInt8>>(4, 3)
        let _ = Vector<StereoSample<Int16>>(count: 1024)

        let m = Matrix<UInt8>(4, 3)
        let a = Vector<Float>(scalars: [1, 2, 3, 4])
        let b = Vector<Float>(scalars: [4, 3, 2, 1])
        let y = Vector<Float>(scalars: [0, 1, 2, 3])
        let expected = Vector<Float>(scalars: [5, 5, 5, 5])

        do {
            //---------------------------------
            // unscoped, uses Platform.defaultStream
            let c1 = a + b
            XCTAssert(c1 == expected)

            //---------------------------------
            // other scoped
            // create a streams on each specified devices from the preferred service
            let stream = try Platform.local.createStreams(deviceIds: [0, 1])
            
            let c3 = using(stream[0]) {
                a + b
            }
            XCTAssert(c3 == expected)
            
            //---------------------------------
            // sequential multi scoped
            let aPlusB = using(stream[0]) {
                a + b
            }
            
            let aMinusB = using(stream[1]) {
                a - b
            }
            
            let c4 = aPlusB + aMinusB
            let c4expected = Vector<Float>(scalars: [2, 4, 6, 8])
            // all three streams auto sync at this point
            XCTAssert(c4 == c4expected)

            //---------------------------------
            // syntax variations
            // transparent type conversion if the Scalar is compatible
            let _ = a.pow(3)
            let _ = a.pow(3.5)
            // integer array
            let _ = m.pow(2) // okay
            // let _ = try m.pow(2.5)  Ints raised to float power won't compile
            let _ = a + 1    // floats + broadcast Int is okay
            let _ = m + 1
            // let _ = try m + 1.5  won't compile
            
            //---------------------------------
            // nested multi scoped
            let x: Vector<Float> = using(stream[0]) {
                let aMinusB = using(stream[1]) {
                    a - b
                }
                // here stream[1] is synced with stream[0]
                return pow(a + b + aMinusB, y)
            }
            
            // can compose easily (but log(x) is computed twice in this case)
            let _ = x.log() / (x.log() + 1)
            
            // temporary results are okay, they won't cause data movement
            let logx = log(x)
            
            // here stream[0] is synced with the currentStream
            let c5 = logx / (logx + 1)
            
            // currentStream, stream0, and stream1 all sync at this point
            let c5expected = Vector<Float>(scalars: [0, 0.581, 0.782, 0.862])
            let c5All = c5.approximatelyEqual(to: c5expected).all()

            // here the currentStream is synced with the app thread
            // because we are actually taking a Bool scalar value
            let c5IsEqual = try c5All.scalarValue()
            XCTAssert(c5IsEqual)

        } catch {
            XCTFail(String(describing: error))
        }
    }
}
