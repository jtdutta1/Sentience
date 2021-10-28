from sentience.framework.abstract.standard.encoder_decoder import EncoderDecoderSchema, EncoderDecoderArch

m_schema = EncoderDecoderSchema(40, 10)
print(m_schema.SCHEMA)
arch = EncoderDecoderArch(m_schema, [40])
arch.create_arch()
print(arch.ARCH)
