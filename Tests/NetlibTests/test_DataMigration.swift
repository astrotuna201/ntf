//******************************************************************************
//  Created by Edward Connell on 12/1/16
//  Copyright © 2016 Connell Research. All rights reserved.
//
import XCTest
import Foundation

@testable import Netlib

class test_DataMigration: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_viewMutateOnWrite", test_viewMutateOnWrite),
        ("test_tensorDataMigration", test_tensorDataMigration),
        ("test_mutateOnDevice", test_mutateOnDevice),
        //            ("test_copyOnWriteCrossDevice", test_copyOnWriteCrossDevice),
        //            ("test_copyOnWriteDevice", test_copyOnWriteDevice),
        //            ("test_copyOnWrite", test_copyOnWrite),
        //            ("test_columnMajorDataView", test_columnMajorDataView),
        //            ("test_columnMajorStrides", test_columnMajorStrides),
    ]
	
    //==========================================================================
	// test_viewMutateOnWrite
	func test_viewMutateOnWrite() {
		do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a Matrix and give it an optional name for logging
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
	
    //==========================================================================
    // test_tensorDataMigration
    //
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
            let stream = Platform.local
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

    //==========================================================================
    // test_mutateOnDevice
    func test_mutateOnDevice() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a named stream on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let stream = Platform.local
                .createStreams(serviceName: "cpuUnitTest", deviceIds: [1, 2])
            XCTAssert(stream[0].device.memoryAddressing == .discreet &&
                stream[1].device.memoryAddressing == .discreet)

            // create a Matrix on device 1 and fill with indexes
            // memory is only allocated on device 1. This also shows how a
            // temporary can be used in a scope. No memory is copied.
            var matrix = using(stream[0]) {
                Matrix<Float>(extents: [3, 2]).filledWithIndex()
            }

            // retreive value on app thread
            // memory is allocated in the host app space and the data is copied
            // from device 1 to the host using stream 0.
            let value1 = matrix.value(at: [1, 1])
            XCTAssert(value1 == 3.0)

            // simulate a readonly kernel access on device 1.
            // matrix was not previously modified, so it is up to date
            // and no data movement is necessary
            _ = try matrix.readOnly(using: stream[0])

            // sum device 1 copy, which should equal 15.
            // This `sum` syntax creates a temporary result on device 1,
            // then `scalarValue` causes the temporary to be transferred to
            // the host, the value is retrieved, and the temp is released.
            // This syntax is good for experiments, but should not be used
            // for repetitive actions
            var sum = using(stream[0]) {
                matrix.sum().scalarValue()
            }
            XCTAssert(sum == 15.0)

            // copy the matrix and simulate a readOnly operation on device2
            // a device array is allocated on device 2 then the master copy
            // on device 1 is copied to device 2.
            // Since device 1 and 2 are in the same service, a device to device
            // async copy is performed. In the case of Cuda, it would travel
            // across nvlink and not the PCI bus
            let matrix2 = matrix
            _ = try matrix2.readOnly(using: stream[1])
            
            // copy matrix2 and simulate a readWrite operation on device2
            // this causes copy on write and mutate on device
            var matrix3 = matrix2
            _ = try matrix3.readWrite(using: stream[1])

            // sum device 1 copy should be 15
            // `sum` creates a temp result tensor, allocates an array on
            // device 2, and performs the reduction.
            // Then `scalarValue` causes a host array to be allocated, and the
            // the data is copied from device 2 to host, the value is returned
            // and the temporary tensor is released.
            sum = using(stream[1]) {
                matrix.sum().scalarValue()
            }
            XCTAssert(sum == 15.0)

            // matrix is overwritten with a new array on device 1
            matrix = using(stream[0]) {
                matrix.filledWithIndex()
            }
            
            // sum matrix on device 2
            // `sum` creates a temporary result tensor on device 2
            // a device array for `matrix` is allocated on device 2 and
            // the matrix data is copied from device 1 to device 2
            // then `scalarValue` creates a host array and the result is
            // copied from device 2 to the host array, and then the tensor
            // is released.
            sum = using(stream[1]) {
                matrix.sum().scalarValue()
            }
            XCTAssert(sum == 15.0)

            // exiting the scopy, matrix and matrix2 are released along
            // with all resources on all devices.
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //--------------------------------------------------------------------------
    // test_copyOnWriteDevice
    func test_copyOnWriteDevice() {
        Platform.log.level = .diagnostic
        Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
        
        // create a named stream on two different discreet devices
        // cpu devices 1 and 2 are discreet memory versions for testing
        let stream = Platform.local
            .createStreams(serviceName: "cpuUnitTest", deviceIds: [1, 2])
        XCTAssert(stream[0].device.memoryAddressing == .discreet &&
            stream[1].device.memoryAddressing == .discreet)
        
        // fill with index on device 1
        let index = [1, 1]
        var matrix1 = Matrix<Float>(extents: [3, 2])
        using(stream[0]) {
            fillWithIndex(&matrix1)
        }
        // testing a value causes the data to be copied to the host
        XCTAssert(matrix1.value(at: index) == 3.0)
        
        // copy and mutate data
        // the data will be duplicated wherever the source is
        var matrix2 = matrix1
        XCTAssert(matrix2.value(at: index) == 3.0)
        
        // writing to matrix2 causes view mutation and copy on write
        matrix2.set(value: 7, at: index)
        XCTAssert(matrix1.value(at: index) == 3.0)
        XCTAssert(matrix2.value(at: index) == 7.0)
    }

    //--------------------------------------------------------------------------
    // test_copyOnWriteCrossDevice
    func test_copyOnWriteCrossDevice() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
            // create a named stream on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let stream = Platform.local
                .createStreams(serviceName: "cpuUnitTest", deviceIds: [1, 2])
            XCTAssert(stream[0].device.memoryAddressing == .discreet &&
                stream[1].device.memoryAddressing == .discreet)

            let index = [1, 1]
            var matrix1 = Matrix<Float>(extents: [3, 2])
            using(stream[0]) {
                fillWithIndex(&matrix1)
            }
            // testing a value causes the data to be copied to the host
            XCTAssert(matrix1.value(at: index) == 3.0)

            // simulate read only access on device 1 and 2
            _ = try matrix1.readOnly(using: stream[0])
            _ = try matrix1.readOnly(using: stream[1])

            // sum device 1 copy should be 15
            let sum1 = using(stream[0]) {
                matrix1.sum().scalarValue()
            }
            XCTAssert(sum1 == 15.0)

            // clear the device 0 master copy
            using(stream[0]) {
                fill(&matrix1, with: 0)
            }

            // sum device 1 copy should now also be 0
            // sum device 1 copy should be 15
            let sum2 = using(stream[1]) {
                matrix1.sum().scalarValue()
            }
            XCTAssert(sum2 == 0)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //--------------------------------------------------------------------------
    // test_copyOnWrite
    // NOTE: uses the default stream
    func test_copyOnWrite() {
        Platform.log.level = .diagnostic
        Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
        
        let index = [1, 1]
        var matrix1 = Matrix<Float>(extents: [3, 2])
        fillWithIndex(&matrix1)
        XCTAssert(matrix1.value(at: index) == 3.0)
        
        var matrix2 = matrix1
        XCTAssert(matrix2.value(at: index) == 3.0)
        matrix2.set(value: 7, at: index)
        XCTAssert(matrix1.value(at: index) == 3.0)
        XCTAssert(matrix2.value(at: index) == 7.0)
    }

    //--------------------------------------------------------------------------
    // test_columnMajorDataView
    // NOTE: uses the default stream
    func test_columnMajorDataView() {
        let cmArray: [Int32] = [0, 3, 1, 4, 2, 5]
        let cmMatrix = Matrix<Int32>(extents: [3, 2],
                                     isColMajor: true,
                                     scalars: cmArray)
        let expected = (0..<cmMatrix.shape.elementCount).map { Int32($0) }
        let rowMajorValues = [Int32](cmMatrix.values())
        XCTAssert(rowMajorValues == expected, "values don't match")
        
        // create row major view from cmData, this will copy and reorder
        let rmMatrix = Matrix<Int32>(extents: [3, 2], scalars: cmArray.values())
        let rowMajorValues2 = [Int32](rmMatrix.values())
        XCTAssert(rowMajorValues2 == expected, "values don't match")
    }
}
