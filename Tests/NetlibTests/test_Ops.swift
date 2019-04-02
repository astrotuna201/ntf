//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
//import TensorFlow
@testable import Netlib
//@testable import DeepLearning

class test_Ops: XCTestCase {
    static var allTests = [
        ("test_Casting", test_Casting),
        ("test_GetDefaultStream", test_GetDefaultStream),
        ("test_PrimaryOps", test_PrimaryOps),
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
    
    func test_PrimaryOps() {
        let a = VectorTensor<Float>(scalars: [1, 2, 3, 4])
        let b = VectorTensor<Float>(scalars: [4, 3, 2, 1])
        let y = VectorTensor<Float>(scalars: [0, 1, 2, 3])
        let expected = VectorTensor<Float>(scalars: [5, 5, 5, 5])

        do {
            //---------------------------------
            // unscoped, uses Platform.defaultStream
            let c1 = a + b
            XCTAssert(c1 == expected)
            XCTAssert(_ThreadLocal.value.noError)
            
            //---------------------------------
            // default stream scoped, uses Platform.defaultStream
            let c2 = try usingDefaultStream {
                return a + b
            }
            XCTAssert(c2 == expected)

            //---------------------------------
            // other scoped
            // create a streams on each specified devices from the preferred service
            let stream = try Platform.global.createStreams(deviceIds: [0, 1])
            
            let c3 = try using(stream[0]) {
                return a + b
            }
            XCTAssert(c3 == expected)

            //---------------------------------
            // sequential multi scoped
            let aPlusB = try using(stream[0]) {
                return a + b
            }
            
            let aMinusB = try using(stream[1]) {
                return a - b
            }
            
            let c4 = try usingDefaultStream {
                return aPlusB + aMinusB
            }
            let c4expected = VectorTensor<Float>(scalars: [2, 4, 6, 8])
            // all three streams auto sync at this point
            XCTAssert(c4 == c4expected)

            //---------------------------------
            // nested multi scoped
            let c5: VectorTensor<Float> = try usingDefaultStream {
                let x: VectorTensor<Float> = try using(stream[0]) {
                    let aMinusB = try using(stream[1]) {
                        return a - b
                    }
                    // here stream[1] is synced with stream[0]
                    return try pow(a + b + aMinusB, y)
                }
                // temporary results are okay!
                // here stream[0] is synced with the defaultStream
                let lnx = try log(x)
                return lnx / (lnx + 1)
            }
            
            // all three streams auto sync at this point
            let c5expected = VectorTensor<Float>(scalars: [0, 0.581, 0.782, 0.862])
            let c5IsEqual = try c5.elementsApproximatelyEqual(c5expected).value()
            
            // here the defaultStream is synced with the app thread
            XCTAssert(c5IsEqual)

        } catch {
            XCTFail(String(describing: error))
        }
    }
}
