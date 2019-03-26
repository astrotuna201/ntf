//
//  CodableTests.swift
//  NetlibTests
//
//  Created by Edward Connell on 3/23/19.
//
import XCTest
import Foundation
import TensorFlow
@testable import Netlib
@testable import DeepLearning

class test_Ops: XCTestCase {
    static var allTests = [
        ("test_PrimaryOps", test_PrimaryOps),
    ]
    
    func test_PrimaryOps() {
        let a = TensorView<Float>(scalars: [1, 2, 3, 4])
        let b = TensorView<Float>(scalars: [4, 3, 2, 1])
        let y = TensorView<Float>(scalars: [0, 1, 2, 3])
        let expected = TensorView<Float>(scalars: [5, 5, 5, 5])

        do {
            //---------------------------------
            // unscoped
            let c1 = a + b
            XCTAssert(c1 == expected)
            XCTAssert(_ThreadLocal.value.noError)
            
            //---------------------------------
            // default scoped
            let c2 = try usingDefaultStream {
                return a + b
            }
            XCTAssert(c2 == expected)

            //---------------------------------
            // other scoped
            // create 2 streams on the specified devices from the preferred service
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
            // all three streams auto sync at this point
            let c4expected = TensorView<Float>(scalars: [2, 4, 6, 8])
            XCTAssert(c4 == c4expected)

            //---------------------------------
            // nested multi scoped
            // NOTE: for some reason I get this when nesting, any thoughts?
            // "Unable to infer complex closure return type; add explicit type to disambiguate"
            //
            let c5 = try usingDefaultStream { () -> TensorView<Float> in
                let x = try using(stream[0]) { () -> TensorView<Float> in
                    let aMinusB = try using(stream[1]) {
                        return a - b
                    }
                    return pow(a + b + aMinusB, y)
                }
                let lnx = log(x)
                return lnx / (lnx + 1)
            }
            // all three streams auto sync at this point
            let c5expected = TensorView<Float>(scalars: [0, 0.581, 0.782, 0.862])
            let c5IsEqual = try c5.elementsApproximatelyEqual(c5expected).scalarized()
            XCTAssert(c5IsEqual)

        } catch {
            
        }
    }
}
