//******************************************************************************
//  Created by Edward Connell on 3/23/19
//  Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

public protocol BinaryEncodable: Encodable {
    func binaryEncode(to encoder: BinaryEncoder) throws
}

public protocol BinaryDecodable: Decodable {
    init(fromBinary decoder: BinaryDecoder) throws
}

public typealias BinaryCodable = BinaryEncodable & BinaryDecodable

public protocol CodableStorage: class {
    
}

//==============================================================================
// BinaryEncoder
public class BinaryEncoder: Encoder {
    public let storage: CodableStorage
    public let codingPath: [CodingKey]
    public let userInfo: [CodingUserInfoKey : Any]
    
    public init(storage: CodableStorage,
                codingPath: [CodingKey] = [],
                userInfo: [CodingUserInfoKey : Any] = [:]) {
        // init
        self.storage = storage
        self.codingPath = codingPath
        self.userInfo = userInfo
    }

    public func container<Key>(keyedBy type: Key.Type) -> KeyedEncodingContainer<Key> where Key: CodingKey {
        fatalError()
    }
    
    public func unkeyedContainer() -> UnkeyedEncodingContainer {
        fatalError()
    }
    
    public func singleValueContainer() -> SingleValueEncodingContainer {
        fatalError()
    }
    
    public func encode<T>(_ value: T) throws -> Data where T: BinaryEncodable {
        fatalError()
    }
}

//==============================================================================
// BinaryEncoder
public class BinaryDecoder: Decoder {
    public var codingPath: [CodingKey]
    public var userInfo: [CodingUserInfoKey : Any]
    
    public init(codingPath: [CodingKey], userInfo: [CodingUserInfoKey : Any] = [:]) {
        self.codingPath = codingPath
        self.userInfo = userInfo
    }

    public func container<Key>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> where Key : CodingKey {
        fatalError()
    }
    
    public func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        fatalError()
    }
    
    public func singleValueContainer() throws -> SingleValueDecodingContainer {
        fatalError()
    }
}
