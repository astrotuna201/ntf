//******************************************************************************
//  Created by Edward Connell on 12/1/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_TensorView: XCTestCase {
    static var allTests : [(String, (test_TensorView) -> () throws -> Void)] {
        return [
            ("test_viewMutateOnWrite", test_viewMutateOnWrite),
            ("test_tensorDataMigration", test_tensorDataMigration),
//            ("test_mutateOnDevice", test_mutateOnDevice),
//            ("test_copyOnWriteCrossDevice", test_copyOnWriteCrossDevice),
//            ("test_copyOnWriteDevice", test_copyOnWriteDevice),
//            ("test_copyOnWrite", test_copyOnWrite),
//            ("test_columnMajorDataView", test_columnMajorDataView),
//            ("test_columnMajorStrides", test_columnMajorStrides),
        ]
    }
	
	//--------------------------------------------------------------------------
	// test_viewMutateOnWrite
	func test_viewMutateOnWrite() {
		do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a Matrix and give it a name for logging
            let values = (0..<12).map { Float($0) }
            var m0 = Matrix<Float>(extents: [3, 4],
                                   name: "weights",
                                   scalars: values)
            let _ = try m0.readWrite()
            XCTAssert(!m0.lastAccessMutatedView)
            let _ = try m0.readOnly()
            XCTAssert(!m0.lastAccessMutatedView)
            let _ = try m0.readWrite()
            XCTAssert(!m0.lastAccessMutatedView)
            
            // copy the view
            var m1 = m0
            // rw access m0 should mutate m0
            let _ = try m0.readWrite()
            XCTAssert(m0.lastAccessMutatedView)
            // m1 should now be unique reference
            XCTAssert(m1.isUniqueReference())
            let _ = try m1.readOnly()
            XCTAssert(!m1.lastAccessMutatedView)

            // copy the view
            var m2 = m0
            let _ = try m2.readOnly()
            XCTAssert(!m2.lastAccessMutatedView)
            // rw request should cause copy of m0 data
            let _ = try m2.readWrite()
            XCTAssert(m2.lastAccessMutatedView)
            // m2 should now be unique reference
            XCTAssert(m2.isUniqueReference())
            
        } catch {
			XCTFail(String(describing: error))
		}
	}
	
    //--------------------------------------------------------------------------
    // test_tensorDataMigration
    // This test uses the default UMA cpu service stream, combined with the
    // cpuUnitTest service, using 2 discreet memory device streams.
    // The purpose is to test data replication and synchronization in the
    // following combinations.
    //
    // `app` means app thread
    // `uma` means any device that shares memory with the app thread
    // `discreet` is any device that does not share memory
    // `same service` means moving data within (cuda gpu:0 -> cuda gpu:1)
    // `cross service` means moving data between services
    //                 (cuda gpu:1 -> gcp tpu:0)
    //
    func test_tensorDataMigration() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a named stream on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let stream = try Platform.local
                .createStreams(serviceName: "cpuUnitTest", deviceIds: [1, 2])
            XCTAssert(stream[0].device.memoryAddressing == .discreet &&
                stream[1].device.memoryAddressing == .discreet)
            
            // create a tensor and validate migration
            let values = (0..<24).map { Float($0) }
            var view = Volume<Float>(extents: [2, 3, 4], scalars: values)
            
            _ = try view.readOnly()
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            _ = try view.readOnly()
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            // this device is not UMA so it
            // ALLOC device array on cpu:1
            // COPY  host --> cpu:1_s0
            _ = try view.readOnly(using: stream[0])
            XCTAssert(view.tensorData.lastAccessCopiedBuffer)

            // write access hasn't been taken, so this is still up to date
            _ = try view.readOnly()
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            // an up to date copy is already there, so won't copy
            _ = try view.readWrite(using: stream[0])
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            // ALLOC device array on cpu:1
            // COPY  cpu:1 --> cpu:2_s0
            _ = try view.readOnly(using: stream[1])
            XCTAssert(view.tensorData.lastAccessCopiedBuffer)
            
            _ = try view.readOnly(using: stream[0])
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            _ = try view.readOnly(using: stream[1])
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            _ = try view.readWrite(using: stream[0])
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            // the master is on cpu:1 so we need to update cpu:2's version
            // COPY cpu:1 --> cpu:2_s0
            _ = try view.readOnly(using: stream[1])
            XCTAssert(view.tensorData.lastAccessCopiedBuffer)
            
            _ = try view.readWrite(using: stream[1])
            XCTAssert(!view.tensorData.lastAccessCopiedBuffer)

            // the master is on cpu:2 so we need to update cpu:1's version
            // COPY cpu:2 --> cpu:1_s0
            _ = try view.readWrite(using: stream[0])
            XCTAssert(view.tensorData.lastAccessCopiedBuffer)
            
            // the master is on cpu:1 so we need to update cpu:2's version
            // COPY cpu:1 --> cpu:2_s0
            _ = try view.readWrite(using: stream[1])
            XCTAssert(view.tensorData.lastAccessCopiedBuffer)
            
            // accessing data without a stream causes transfer to the host
            // COPY cpu:2_s0 --> host
            _ = try view.readOnly()
            XCTAssert(view.tensorData.lastAccessCopiedBuffer)

        } catch {
            XCTFail(String(describing: error))
        }
    }

    //----------------------------------------------------------------------------
//    // test_mutateOnDevice
//    func test_mutateOnDevice() {
//        do {
//            let model = Model()
//            try model.setup()
//            let stream = try model.compute.requestStreams(label: "dataStream", deviceIds: [0, 1])
//
//            var data0 = DataView(rows: 3, cols: 2)
//            try stream[0].fillWithIndex(data: &data0, startingAt: 0)
//
//            let value1: Float = try data0.get(at: [1, 1])
//            XCTAssert(value1 == 3.0)
//
//            // migrate the data to the devices
//            _ = try data0.ro(using: stream[0])
//
//            // sum device 0 copy should be 15
//            var sum = DataView(count: 1)
//            try stream[0].asum(x: data0.flattened(), result: &sum)
//            var sumValue: Float = try sum.get()
//            XCTAssert(sumValue == 15.0)
//
//            let data1 = data0
//            _ = try data1.ro(using: stream[1])
//
//            // sum device 1 copy should be 15
//            try stream[1].asum(x: data0.flattened(), result: &sum)
//            sumValue = try sum.get()
//            XCTAssert(sumValue == 15.0)
//
//            // clear stream 0 copy
//            try stream[0].fill(data: &data0, with: 0)
//
//            // sum device 1 copy should still be 15
//            try stream[1].asum(x: data1.flattened(), result: &sum)
//            sumValue = try sum.get()
//            XCTAssert(sumValue == 15.0)
//            //            print(sumValue)
//
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //----------------------------------------------------------------------------
//    // test_copyOnWriteDevice
//    func test_copyOnWriteDevice() {
//        do {
//            let model = Model()
//            try model.setup()
//            let stream = try model.compute.requestStreams(label: "dataStream")[0]
//
//            let testIndex = [1, 1]
//            var data1 = DataView(rows: 3, cols: 2)
//            try cpuFillWithIndex(data: &data1, startingAt: 0)
//            let value1: Float = try data1.get(at: testIndex)
//            XCTAssert(value1 == 3.0)
//
//            // migrate the data to the device
//            _ = try data1.ro(using: stream)
//
//            // copy and mutate data
//            // the data will be duplicated wherever the source is
//            var data2 = data1
//            let value2: Float = try data2.get(at: testIndex)
//            XCTAssert(value2 == 3.0)
//            try data2.set(value: 7, at: [1, 1])
//
//            let value1a: Float = try data1.get(at: testIndex)
//            XCTAssert(value1a == 3.0)
//
//            let value2a: Float = try data2.get(at: testIndex)
//            XCTAssert(value2a == 7.0)
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //----------------------------------------------------------------------------
//    // test_copyOnWriteCrossDevice
//    func test_copyOnWriteCrossDevice() {
//        do {
//            let model = Model()
//            try model.setup()
//            let stream = try model.compute.requestStreams(label: "dataStream", deviceIds: [0, 1])
//            let multiDevice = stream[0].device.id != stream[1].device.id
//
//            // don't test unless we have multiple devices
//            if !multiDevice { return }
//
//            let testIndex = [0, 0, 1, 1]
//            var data1 = DataView(rows: 3, cols: 2)
//            try stream[0].fillWithIndex(data: &data1, startingAt: 0)
//            let value1: Float = try data1.get(at: testIndex)
//            XCTAssert(value1 == 3.0)
//
//            // migrate the data to the devices
//            _ = try data1.ro(using: stream[0])
//            _ = try data1.ro(using: stream[1])
//
//            // sum device 0 copy should be 15
//            var sum = DataView(count: 1)
//            try stream[0].asum(x: data1.flattened(), result: &sum)
//            var sumValue: Float = try sum.get()
//            XCTAssert(sumValue == 15.0)
//
//            // clear the device 0 master copy
//            try stream[0].fill(data: &data1, with: 0)
//
//            // sum device 1 copy should now also be 0
//            try stream[1].asum(x: data1.flattened(), result: &sum)
//            sumValue = try sum.get()
//            XCTAssert(sumValue == 0.0)
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //----------------------------------------------------------------------------
//    // test_copyOnWrite
//    func test_copyOnWrite() {
//        do {
//            let testIndex = [1, 1]
//            var data1 = DataView(rows: 3, cols: 2)
//            try cpuFillWithIndex(data: &data1, startingAt: 0)
//            let value1: Float = try data1.get(at: testIndex)
//            XCTAssert(value1 == 3.0)
//
//            var data2 = data1
//            let value2: Float = try data2.get(at: testIndex)
//            XCTAssert(value2 == 3.0)
//            try data2.set(value: 7, at: testIndex)
//
//            let value1a: Float = try data1.get(at: testIndex)
//            XCTAssert(value1a == 3.0)
//
//            let value2a: Float = try data2.get(at: testIndex)
//            XCTAssert(value2a == 7.0)
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
//
//    //----------------------------------------------------------------------------
//    // test_columnMajorStrides
//    func test_columnMajorStrides() {
//        let extent_nchw = [1, 2, 3, 4]
//        let rmShape4 = Shape(extent: extent_nchw)
//        XCTAssert(rmShape4.strides == [24, 12, 4, 1])
//
//        let cmShape4 = Shape(extent: extent_nchw, colMajor: true)
//        XCTAssert(cmShape4.strides == [24, 12, 1, 3])
//
//        let extent_nhwc = [1, 3, 4, 2]
//        let irmShape4 = Shape(extent: extent_nhwc, layout: .nhwc)
//        XCTAssert(irmShape4.strides == [24, 8, 2, 1])
//
//        let icmShape4 = Shape(extent: extent_nhwc, layout: .nhwc, colMajor: true)
//        XCTAssert(icmShape4.strides == [24, 2, 6, 1])
//
//        //-------------------------------------
//        // rank 5
////        let extent5  = Extent(10, 3, 4, 2, 3)
////        let rmShape5 = Shape(extent: extent5)
////        XCTAssert(rmShape5.strides[0] == 72)
////        XCTAssert(rmShape5.strides[1] == 24)
////        XCTAssert(rmShape5.strides[2] == 6)
////        XCTAssert(rmShape5.strides[3] == 3)
////        XCTAssert(rmShape5.strides[4] == 1)
////
////        // rank 5
////        let irmShape5 = Shape(extent: extent5, isInterleaved: true)
////        XCTAssert(irmShape5.strides[0] == 72)
////        XCTAssert(irmShape5.strides[1] == 1)
////        XCTAssert(irmShape5.strides[2] == 18)
////        XCTAssert(irmShape5.strides[3] == 9)
////        XCTAssert(irmShape5.strides[4] == 3)
////
////        // rank 5
////        let cmShape5 = Shape(extent: extent5, isColMajor: true)
////        XCTAssert(cmShape5.strides[0] == 72)
////        XCTAssert(cmShape5.strides[1] == 24)
////        XCTAssert(cmShape5.strides[2] == 6)
////        XCTAssert(cmShape5.strides[3] == 1)
////        XCTAssert(cmShape5.strides[4] == 2)
////
////        // rank 5
////        let icmShape5 = Shape(extent: extent5, isInterleaved: true, isColMajor: true)
////        XCTAssert(icmShape5.strides[0] == 72)
////        XCTAssert(icmShape5.strides[1] == 1)
////        XCTAssert(icmShape5.strides[2] == 18)
////        XCTAssert(icmShape5.strides[3] == 3)
////        XCTAssert(icmShape5.strides[4] == 6)
//    }
//
//    //----------------------------------------------------------------------------
//    // test_columnMajorDataView
//    func test_columnMajorDataView() {
//        do {
//            // load linear buffer with values in col major order
//            let cmArray: [UInt8] = [0, 3, 1, 4, 2, 5]
//            let extent  = [2, 3]
//            let cmShape = Shape(extent: extent, colMajor: true)
//
//            // create a data view
//            var cmData   = DataView(shape: cmShape, dataType: .real8U)
//            let cmBuffer = try cmData.rwReal8U()
//            for i in 0..<cmArray.count { cmBuffer[i] = cmArray[i] }
//
//            // test col major indexing
//            var i: UInt8 = 0
//            for row in 0..<cmData.rows {
//                for col in 0..<cmData.cols {
//                    let value: UInt8 = try cmData.get(at: [row, col])
//                    XCTAssert(value == i)
//                    i += 1
//                }
//            }
//
//            // create row major view from cmData, this will copy and reorder
//            let rmShape = Shape(extent: extent)
//            let rmData = try DataView(from: cmData, asShape: rmShape)
//            let rmBuffer = try rmData.roReal8U()
//            for i in 0..<rmData.elementCount { XCTAssert(rmBuffer[i] == UInt8(i))    }
//
//            // test row major indexing
//            i = 0
//            for row in 0..<rmData.rows {
//                for col in 0..<rmData.cols {
//                    let value: UInt8 = try rmData.get(at: [row, col])
//                    XCTAssert(value == i)
//                    i += 1
//                }
//            }
//        } catch {
//            XCTFail(String(describing: error))
//        }
//    }
}
