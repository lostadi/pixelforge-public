package org.ostadix.nano;

import android.os.IBinder;
import android.os.Parcel;
import android.os.ParcelFileDescriptor;
import android.os.SharedMemory;
import android.system.Os;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;

/** Owned reconstruction of the installed factory-metadata reader's Binder ABI. */
public final class FactoryReader {
    private static final String SERVICE = "vendor.google.plat_security.ITrustyDecrypt/default";
    private static final String DESCRIPTOR = "vendor.google.plat_security.ITrustyDecrypt";

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            throw new IllegalArgumentException("Usage: FactoryReader metadata-filename output-file");
        }
        String name = args[0];
        if (!name.matches("[A-Za-z0-9][A-Za-z0-9_.-]*")) {
            throw new IllegalArgumentException("Expected a single factory filename");
        }
        File input = new File("/data/vendor/intelligence", name);
        if (!input.isFile() || !input.getCanonicalFile().getParentFile()
                .equals(new File("/data/vendor/intelligence").getCanonicalFile())) {
            throw new IllegalArgumentException("Input must be inside the factory store");
        }
        long size = input.length();
        if (size < 4101 || size >= 104857600L || size % 4096 != 0) {
            throw new IllegalArgumentException("Unexpected factory metadata size: " + size);
        }
        File output = new File(args[1]);
        if (output.exists()) throw new IllegalArgumentException("Output already exists");
        System.out.println("event=start uid=" + android.os.Process.myUid()
                + " file=" + name + " input_bytes=" + size);
        IBinder service = (IBinder) Class.forName("android.os.ServiceManager")
                .getMethod("getService", String.class).invoke(null, SERVICE);
        if (service == null) throw new IllegalStateException("TrustyDecrypt service unavailable");
        System.out.println("event=service_found descriptor=" + service.getInterfaceDescriptor());
        if (!DESCRIPTOR.equals(service.getInterfaceDescriptor())) {
            throw new IllegalStateException("Unexpected Binder descriptor");
        }
        ByteBuffer mapped = null;
        try (ParcelFileDescriptor source = ParcelFileDescriptor.open(input,
                     ParcelFileDescriptor.MODE_READ_ONLY);
             SharedMemory memory = SharedMemory.create("ostadix_factory_metadata", (int) size)) {
            mapped = memory.mapReadOnly();
            Parcel sharedParcel = Parcel.obtain();
            ParcelFileDescriptor destination;
            try {
                memory.writeToParcel(sharedParcel, 0);
                sharedParcel.setDataPosition(0);
                destination = sharedParcel.readFileDescriptor();
            } finally {
                sharedParcel.recycle();
            }
            if (destination == null) throw new IllegalStateException("No shared-memory descriptor");
            int status;
            try (ParcelFileDescriptor ownedDestination = destination) {
                Parcel request = Parcel.obtain();
                Parcel reply = Parcel.obtain();
                try {
                    request.writeInterfaceToken(DESCRIPTOR);
                    request.writeInt(1);
                    source.writeToParcel(request, 0);
                    request.writeInt(1);
                    ownedDestination.writeToParcel(request, 0);
                    if (!service.transact(1, request, reply, 0)) {
                        throw new IllegalStateException("Decrypt transaction not implemented");
                    }
                    reply.readException();
                    status = reply.readInt();
                } finally {
                    request.recycle();
                    reply.recycle();
                }
            }
            System.out.println("event=decrypt_return status=" + status);
            if (status != 0) throw new IllegalStateException("TrustyDecrypt failed: " + status);
            int length = mapped.order(ByteOrder.LITTLE_ENDIAN).getInt(mapped.capacity() - 4100);
            if (length <= 0 || length > mapped.capacity() - 4100) {
                throw new IllegalStateException("Invalid decrypted footer length: " + length);
            }
            mapped.position(0);
            mapped.limit(length);
            if (!output.createNewFile()) throw new IllegalStateException("Output creation failed");
            Os.chmod(output.getAbsolutePath(), 0600);
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] chunk = new byte[65536];
            try (FileOutputStream stream = new FileOutputStream(output)) {
                while (mapped.hasRemaining()) {
                    int count = Math.min(chunk.length, mapped.remaining());
                    mapped.get(chunk, 0, count);
                    stream.write(chunk, 0, count);
                    digest.update(chunk, 0, count);
                }
            }
            StringBuilder hash = new StringBuilder();
            for (byte b : digest.digest()) {
                hash.append(String.format("%02x", b & 255));
            }
            System.out.println("event=metadata_written bytes=" + length + " sha256=" + hash);
        } finally {
            if (mapped != null) SharedMemory.unmap(mapped);
        }
    }
}
