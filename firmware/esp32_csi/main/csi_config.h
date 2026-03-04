#ifndef CSI_CONFIG_H
#define CSI_CONFIG_H

#define CSI_WIFI_SSID           "CSI_SENTINEL_AP"
#define CSI_WIFI_PASS           "sentinel2024"
#define CSI_WIFI_CHANNEL        6
#define CSI_SAMPLE_RATE_HZ      200
#define CSI_SAMPLE_INTERVAL_US  (1000000 / CSI_SAMPLE_RATE_HZ)

#define UDP_SERVER_IP           "192.168.4.2"
#define UDP_SERVER_PORT         5500
#define UDP_BUFFER_SIZE         1460

#define CSI_SUBCARRIER_COUNT    64
#define CSI_LLTF_SUBCARRIERS    52
#define CSI_HT_LTF_SUBCARRIERS  56

#define DEVICE_ROLE_TX          0
#define DEVICE_ROLE_RX          1

#define PACKET_MAGIC            0xCSI5
#define PACKET_VERSION          0x01

typedef struct __attribute__((packed)) {
    uint16_t magic;
    uint8_t version;
    uint8_t device_id;
    uint32_t timestamp_us;
    uint16_t sequence;
    int8_t rssi;
    uint8_t noise_floor;
    uint8_t channel;
    uint8_t secondary_channel;
    uint16_t num_subcarriers;
    int8_t csi_data[128];
} csi_packet_t;

#endif
