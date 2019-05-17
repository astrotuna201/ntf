//******************************************************************************
//  Created by Edward Connell on 4/26/19
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_Async: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_StreamEventWait", test_StreamEventWait),
        ("test_perfCreateStreamEvent", test_perfCreateStreamEvent),
    ]

    //==========================================================================
    // test_defaultStreamOp
    // initializes two matrices and adds them together
    func test_bug() {
//        let qv = QVector<UInt8, Float>(sequence: 0..<5)
//        let val = qv.quantizer.convert(viewed: 0.5)
//        print(val)
//
//        let pixv = QVector<RGBSample<UInt8>, RGBSample<Float>>(count: 1)
//        let pval = pixv.quantizer.convert(viewed: RGBSample(r: 0, g: 0.5, b: 1.0))
//        print(pval)
//        do {
//            Platform.log.level = .diagnostic
//            
//            let values: [Float] = [0, -1.5, 2, -3, 4, 5]
//            let m1 = Matrix<Float>((2, 3), values: values)
////            var results = try m1.mutableValues()
////            results[results.startIndex] = 7.5
////            let a = [Float](results)
////            print(a)
////            for i in results.indices {
////                results[i] = 7
////            }
////            let a = [Float](results)
////            print(a)
//
//            let result = abs(m1)
//
//            let values = try result.array()
//            
//            let expected: [Float] = [0, 1.5, 2, 3, 4, 5]
//            XCTAssert(values == expected)
//        } catch {
//            XCTFail(String(describing: error))
//        }
//        
//        if ObjectTracker.global.hasUnreleasedObjects {
//            XCTFail(ObjectTracker.global.getActiveObjectReport())
//        }
    }
    

    //==========================================================================
    // test_defaultStreamOp
    // initializes two matrices and adds them together
    func test_defaultStreamOp() {
        do {
            Platform.log.level = .diagnostic
            
            let m1 = Matrix<Int32>((2, 5), name: "m1", sequence: 0..<10)
            let m2 = Matrix<Int32>((2, 5), name: "m2", sequence: 0..<10)
            let result = m1 + m2
            let values = try result.array()
            
            let expected = (0..<10).map { Int32($0 * 2) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_secondaryDiscreetMemoryStream
    // initializes two matrices on the app thread, executes them on `stream1`,
    // the retrieves the results
    func test_secondaryDiscreetMemoryStream() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .scheduling, .streamSync]

            // create a named stream on a discreet device
            // cpuUnitTest device 1 is a discreet memory versions for testing
            let stream1 = Platform.local
                .createStream(deviceId: 1, serviceName: "cpuUnitTest")
            
            let m1 = Matrix<Int32>((2, 5), name: "m1", sequence: 0..<10)
            let m2 = Matrix<Int32>((2, 5), name: "m2", sequence: 0..<10)

            // perform on user provided discreet memory stream
            let result = using(stream1) { m1 + m2 }

            // synchronize with host stream and retrieve result values
            let values = try result.array()
            
            let expected = (0..<10).map { Int32($0 * 2) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_threeStreamInterleave
    func test_threeStreamInterleave() {
        do {
            Platform.log.level = .diagnostic
            
            // create named streams on two discreet devices
            // cpuUnitTest device 1 and 2 are discreet memory versions
            let stream1 = Platform.local
                .createStream(deviceId: 1, serviceName: "cpuUnitTest")
            let stream2 = Platform.local
                .createStream(deviceId: 2, serviceName: "cpuUnitTest")

            let m1 = Matrix<Int32>((2, 3), name: "m1", sequence: 0..<6)
            let m2 = Matrix<Int32>((2, 3), name: "m2", sequence: 0..<6)
            let m3 = Matrix<Int32>((2, 3), name: "m3", sequence: 0..<6)

            // sum the values with a delay on device 1
            let sum_m1m2: Matrix<Int32> = using(stream1) {
                delayStream(atLeast: 0.1)
                return m1 + m2
            }

            // multiply the values on device 2
            let result = using(stream2) {
                sum_m1m2 * m3
            }

            // synchronize with host stream and retrieve result values
            let values = try result.array()
            
            let expected = (0..<6).map { Int32(($0 + $0) * $0) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_tensorReferenceBufferSync
    func test_tensorReferenceBufferSync() {
    }

    //==========================================================================
    // test_temporaryStreamShutdown
    func test_temporaryStreamShutdown() {
        do {
            Platform.log.level = .diagnostic
            
            for i in 0..<1000 {
                // create a matrix without any storage allocated
                var matrix = Matrix<Int32>((3, 4))
                
                // create a stream just for this closure
                // it will probably try to deinit before `fillWithIndex` is
                // complete
                using(Platform.local.createStream()) {
                    fillWithIndex(&matrix)
                }
                
                // synchronize with host stream and retrieve result values
                let values = try matrix.array()
                let expected = [Int32](0..<12)
                XCTAssert(values == expected, "iteration: \(i) failed")
            }
        } catch {
            XCTFail(String(describing: error))
        }
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }

    //==========================================================================
    // test_StreamEventWait
    func test_StreamEventWait() {
        do {
            Platform.log.level = .diagnostic
            Platform.local.log.categories = [.streamSync]
            
            let stream = Platform.local.createStream(serviceName: "cpuUnitTest")
            let event = try stream.createEvent()
            stream.delayStream(atLeast: 0.001)
            try stream.record(event: event).wait()
            XCTAssert(event.occurred, "wait failed to block")
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_perfCreateStreamEvent
    // measures the event overhead of creating 10,000 events
    func test_perfCreateStreamEvent() {
        #if !DEBUG
        let stream = Platform.local.createStream()
        self.measure {
            do {
                for _ in 0..<10000 {
                    _ = try stream.createEvent()
                }
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }

    //==========================================================================
    // test_perfRecordStreamEvent
    // measures the event overhead of processing 10,000 tensors
    func test_perfRecordStreamEvent() {
        #if !DEBUG
        let stream = Platform.local.createStream()
        self.measure {
            do {
                for _ in 0..<10000 {
                    _ = try stream.record(event: stream.createEvent())
                }
                try stream.waitUntilStreamIsComplete()
            } catch {
                XCTFail(String(describing: error))
            }
        }
        #endif
    }
}
