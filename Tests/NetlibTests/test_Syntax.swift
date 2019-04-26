//******************************************************************************
//  Created by Edward Connell on 4/24/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Syntax: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_appThreadZipMapReduce", test_appThreadZipMapReduce),
        ("test_simple", test_simple),
        ("test_streams", test_streams),
        ("test_structuredScalar", test_structuredScalar),
        ("test_withResultPlacement", test_withResultPlacement),
        ("test_logging", test_logging),
    ]

    //==========================================================================
    // test_appThreadZipMapReduce
    func test_appThreadZipMapReduce() {
        // create two tensors and fill with indexes
        let a = Matrix<Float>((2, 3), sequence: 0..<6)
        let b = Matrix<Float>((2, 3), sequence: 6..<12)
        
        let absum = zip(a, b).map { $0 + $1 }
        
        let absumExpected: [Float] = [6, 8, 10, 12, 14, 16]
        XCTAssert(absum == absumExpected)
        
        let dot = zip(a, b).map(*).reduce(0, +)
        XCTAssert(dot == 145.0)
    }
    
    //==========================================================================
    // test_simple
    func test_simple() {
        //--------------------------------
        // initialize a matrix with a sequence and take the sum
        do {
            let matrix = Matrix<Float>((3, 5), sequence: 0..<15)
            print(matrix.formatted((2,0)))
            let sum = matrix.sum().scalarValue()
            XCTAssert(sum == 105.0)
        }
        
        //--------------------------------
        // zero copy transpose
        do {
            let matrix = Matrix<Float>((3, 5), sequence: 0..<15)
            print(matrix.formatted((2,0)))
            
            let tmatrix = matrix.t
            print(tmatrix.formatted((2,0)))
        }
        
        //--------------------------------
        // Select and sum a 3D sub region
        // - initialize a volume using explicit extents
        // - fill with indexes on the default device
        // - create a sub view and take the sum on the device
        // - return the scalar value back to the app thread
        do {
            let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
            print(volume.formatted((2,0)))
            
            let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
            print(subView.formatted((2,0)))
            
            let subViewSum = sum(subView).scalarValue()
            XCTAssert(subViewSum == 312)
        }
        
        //--------------------------------
        // repeat a value
        // No matter the extents, `volume` only uses the shared storage
        // from `value` and repeats it through indexing
        do {
            let volume = Volume<Int32>((2, 3, 10), repeating: Volume(42))
            print(volume.formatted((2,0)))
        }
        
        //--------------------------------
        // repeat a vector
        // No matter the extents, `matrix` only uses the shared storage
        // from `rowVector` and repeats it through indexing
        do {
            let rowVector = Matrix<Int32>((1, 10), sequence: 0..<10)
            let rmatrix = Matrix((10, 10), repeating: rowVector)
            print(rmatrix.formatted((2,0)))

            let colVector = Matrix<Int32>((10, 1), sequence: 0..<10)
            let cmatrix = Matrix((10, 10), repeating: colVector)
            print(cmatrix.formatted((2,0)))
        }
    }

    //==========================================================================
    // test_streams
    // create a named stream on two different discreet devices
    // <cpu devices 1 and 2 are discreet memory versions for testing>
    func test_streams() {
        let stream1 = Platform.local.createStream(deviceId: 1)
        let stream2 = Platform.local.createStream(deviceId: 2)
        
        let volume = using(stream1) {
            Volume<Int32>((3, 4, 5)).filledWithIndex()
        }
        let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
        
        let subViewSum = using(stream2) {
            sum(subView).scalarValue()
        }
        XCTAssert(subViewSum == 312)
    }

    //==========================================================================
    // test_structuredScalar
    // create a named stream on two different discreet devices
    // <cpu devices 1 and 2 are discreet memory versions for testing>
    func test_structuredScalar() {
        let sample = RGBASample<UInt8>(r: 0, g: 1, b: 2, a: 3)
        let matrix = Matrix<RGBASample<UInt8>>((2, 3),
                                               repeating: Matrix(sample))
        let nhwc = NHWCTensor<UInt8>(matrix)
        print(nhwc.formatted((2, 0)))
    }
    
    //==========================================================================
    // test_withResultPlacement
    func test_withResultPlacement() {
        let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
        let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

        var subViewSum = Volume<Int32>((1, 1, 1))
        sum(subView, result: &subViewSum)
        XCTAssert(subViewSum.scalarValue() == 312)
    }

    //==========================================================================
    // test_logging
    // create a named stream on two different discreet devices
    // <cpu devices 1 and 2 are discreet memory versions for testing>
    func test_logging() {
        Platform.log.level = .diagnostic
        Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

        let stream1 = Platform.local
            .createStream(serviceName: "cpuUnitTest", deviceId: 1)

        let stream2 = Platform.local
            .createStream(serviceName: "cpuUnitTest", deviceId: 2)

        let volume = using(stream1) {
            Volume<Int32>((3, 4, 5)).filledWithIndex()
        }
        let subView = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
        
        let subViewSum = using(stream2) {
            sum(subView).scalarValue()
        }
        XCTAssert(subViewSum == 312)
    }
}
