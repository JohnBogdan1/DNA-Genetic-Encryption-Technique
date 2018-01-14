import string
import random
from time import time

# number of rounds the algorithm is run, chosen randomly
# must be at least 3
rounds_no = random.randint(3, 11)

# generate encoding tables domains
two_bit_list = ['00', '01', '10', '11']
dna_bases = ['A', 'C', 'G', 'T']

four_bit_list = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100',
                 '1101', '1110', '1111']
two_dna_bases = ['TA', 'TC', 'TG', 'TT', 'GA', 'GC', 'GG', 'GT', 'CA', 'CC', 'CG', 'CT', 'AA', 'AC', 'AG', 'AT']

# encoding tables and their reversal
two_bits_to_dna_base_table = None
dna_base_to_two_bits_table = None

four_bits_to_two_dna_base_table = None
two_dna_base_to_four_bits_table = None

chromosome_length = None


def str2bin(sstring):
    """
    Transform a string (e.g. 'Hello') into a string of bits
    """
    bs = ''
    for c in sstring:
        bs = bs + bin(ord(c))[2:].zfill(8)
    return bs


def byte2bin(byte_val):
    """
    Transform a byte (8-bit) value into a bitstring
    """
    return bin(byte_val)[2:].zfill(8)


def bitxor(a, b):
    """
    Xor two bit strings (trims the longer input)
    """
    return "".join([str(int(x) ^ int(y)) for (x, y) in zip(a, b)])


def generate_pre_processing_tables():
    """
    Generate the 2 bits to dna bases encoding table (e.g. '01'->C)
    """
    global two_bits_to_dna_base_table
    global dna_base_to_two_bits_table

    # if you want random table
    # random.shuffle(dna_bases)
    two_bits_to_dna_base_table = dict(zip(two_bit_list, dna_bases))
    dna_base_to_two_bits_table = dict(zip(two_bits_to_dna_base_table.values(), two_bits_to_dna_base_table.keys()))


def generate_mutation_tables():
    """
    Generate the 4 bits to 2 dna bases encoding table (e.g. '0101'->CG)
    """
    global four_bits_to_two_dna_base_table
    global two_dna_base_to_four_bits_table

    # if you want random table
    # random.shuffle(two_dna_bases)
    four_bits_to_two_dna_base_table = dict(zip(four_bit_list, two_dna_bases))
    two_dna_base_to_four_bits_table = dict(
        zip(four_bits_to_two_dna_base_table.values(), four_bits_to_two_dna_base_table.keys()))


def group_bits(byte, step=2):
    """
    Group the bits from a byte / bigger sequence of bits into groups by length "step"
    :return: a list of groups
    """
    bits = []
    for i in range(0, len(byte), step):
        bits.append(byte[i:i + step])
    return bits


def generate_bits(byte_data):
    """
    Take every byte for sequence and group its bits
    :return:
    """
    grouped_bits_data = []

    for byte in byte_data:
        grouped_bits_data.extend(group_bits(byte))

    return grouped_bits_data


def binarized_data(data):
    # convert every char to ASCII and then to binary
    byte_data = [byte2bin(ord(c)) for c in data]

    return generate_bits(byte_data)


def bits_to_dna(data, conversion_table):
    # convert binary sequence to DNA sequence
    return "".join([conversion_table[bits] for bits in data])


def dna_to_bits(data):
    # convert DNA sequence to binary sequence
    return "".join([dna_base_to_two_bits_table[dna_base] for dna_base in data])


def encrypt_key(data, key):
    """
    Encrypt data with key: data XOR key.
    """

    # repeat key ONLY if data is longer than key and encrypt
    if len(data) > len(key):
        factor = int(len(data) / len(key))
        key += key * factor

        return bitxor(data, key)

    return bitxor(data, key)


def reshape(dna_sequence):
    """
    Generate chromosome population.
    :param dna_sequence: a string sequence of DNA bases
    :return: an array of chromosomes, chromosome population
    """
    global chromosome_length

    # choose population size and chromosome length
    chromosome_no = random.randint(2, int(len(dna_sequence) / 2))
    chromosome_length = int(len(dna_sequence) / chromosome_no)
    chromosomes = []

    # retrieve the population
    for i in range(0, len(dna_sequence), chromosome_length):
        chromosomes.append(dna_sequence[i:i + chromosome_length])

    return chromosomes


def reverse_reshape(population):
    # convert the chromosome population back to DNA sequence
    return "".join(population)


def rotate_crossover(population):
    """
    Rotate every chromosome in population left / right according to probability p.
    """
    global chromosome_length

    new_population = []

    # predefined rotation value, varied every round
    rotation_offset = random.randint(0, chromosome_length)

    for chromosome in population:

        p = random.uniform(0, 1)

        if p > 0.5:
            right_first = chromosome[0: len(chromosome) - rotation_offset]
            right_second = chromosome[len(chromosome) - rotation_offset:]
            new_population.append(right_second + right_first)
        else:
            left_first = chromosome[0: rotation_offset]
            left_second = chromosome[rotation_offset:]
            new_population.append(left_second + left_first)

    return new_population


def single_point_crossover(population):
    """
    Combine each two chromosomes in population by using single point crossover.
    """
    new_population = []
    for i in range(0, len(population) - 1, 2):
        candidate1 = population[i]
        candidate2 = population[i + 1]

        # if the length of the two chromosomes differ, get the length of the smallest one
        # choose the crossover_point based on this point
        # else if the length of the two chromosomes don't differ, length is the same for both, doesn't matter
        length = min(len(candidate1), len(candidate2))
        crossover_point = random.randint(0, length - 1)
        offspring1 = candidate2[0: crossover_point] + candidate1[crossover_point:]
        offspring2 = candidate1[0: crossover_point] + candidate2[crossover_point:]
        new_population.append(offspring1)
        new_population.append(offspring2)

    # append last chromosome if odd population size
    if len(population) % 2 == 1:
        new_population.append(population[len(population) - 1])

    return new_population


def crossover(population):
    # choose crossover type according to p
    p = random.uniform(0, 1)

    if p < 0.3:
        return rotate_crossover(population)
    elif p >= 0.3 and p < 0.6:
        return single_point_crossover(population)
    else:
        population = rotate_crossover(population)
        return single_point_crossover(population)


def complement(chromosome, point1, point2):
    """
    Flip chromosome bits between point1 and point2.
    """
    new_chromosome = ""

    for i in range(len(chromosome)):
        if i >= point1 and i <= point2:
            if chromosome[i] == '0':
                new_chromosome += '1'
            else:
                new_chromosome += '0'
        else:
            new_chromosome += chromosome[i]

    return new_chromosome


def alter_dna_bases(bases):
    """
    Alter DNA bases to another one randomly.(e.g. C->G and A->T and viceversa)
    """
    alter_dna_table = {}

    for _ in range(2):
        # choose one randomly then remove it from list
        base1 = bases[random.randint(0, len(bases) - 1)]
        bases.remove(base1)

        # choose one randomly then remove it from list
        base2 = bases[random.randint(0, len(bases) - 1)]
        bases.remove(base2)

        # assign the first to the other
        alter_dna_table[base1] = base2
        alter_dna_table[base2] = base1

    return alter_dna_table


def mutation(population):
    """
    Apply mutation operator by using "complement" and "alter_dna_bases"
    """
    global two_bits_to_dna_base_table
    global four_bits_to_two_dna_base_table

    bases = ['A', 'C', 'G', 'T']
    alter_dna_table = alter_dna_bases(bases)

    new_population = []
    for chromosome in population:
        # apply the complement
        b_chromosome = dna_to_bits(chromosome)
        point1 = random.randint(0, len(b_chromosome) - 1)
        point2 = random.randint(point1, len(b_chromosome) - 1)
        b_chromosome = complement(b_chromosome, point1, point2)

        # convert each 4 bits in chromosome to two dna bases using four_bits_to_two_dna_base_table
        four_bits_vector = group_bits(b_chromosome, 4)

        last_dna_base = None
        # if the last element is of length 2, don't convert it
        if len(four_bits_vector[len(four_bits_vector) - 1]) == 2:
            last_dna_base = two_bits_to_dna_base_table[four_bits_vector[len(four_bits_vector) - 1]]

            # convert only the 4 bits elements
            four_bits_vector = four_bits_vector[:-1]

        dna_seq = bits_to_dna(four_bits_vector, four_bits_to_two_dna_base_table)
        if last_dna_base is not None:
            dna_seq += last_dna_base

        # and then alter the dna bases between point1 and point2
        point1 = random.randint(0, len(dna_seq) - 1)
        point2 = random.randint(point1, len(dna_seq) - 1)
        new_chromosome = ""
        for i in range(len(dna_seq)):
            if i >= point1 and i <= point2:
                new_chromosome += alter_dna_table[dna_seq[i]]
            else:
                new_chromosome += dna_seq[i]

        new_population.append(new_chromosome)

    return new_population


def dna_get(text, key):
    global rounds_no
    global two_bits_to_dna_base_table

    print("\nDNA-GET is running...\n")

    # binarize data and convert it to dna sequence
    b_data1 = binarized_data(text)
    dna_seq = bits_to_dna(b_data1, two_bits_to_dna_base_table)
    # print(dna_seq)

    # there is no need for first reshape like in the pseudocode because my reverse_reshape can work on dna_seq, too
    # i.e. ("".join("ACGT") -> "ACGT")

    b_data2 = dna_seq
    print("Initial DNA sequence:", dna_seq)

    # run the algorithm "rounds_no" times
    while rounds_no > 0:
        # encrypt data with key after reshaping it back to binary sequence and then convert it back to dna sequence
        b_data2 = bits_to_dna(
            group_bits(encrypt_key(dna_to_bits(reverse_reshape(b_data2)), key)), two_bits_to_dna_base_table)
        # print("Encrypted data:", b_data2)

        # create the chromosome population
        b_data2 = reshape(b_data2)
        # print("Population data:", b_data2)

        # apply crossover on population
        b_data2 = crossover(b_data2)
        # print("Population data:", b_data2)

        # apply mutation on population
        b_data2 = mutation(b_data2)
        # print("Population data:", b_data2)

        rounds_no -= 1

    return reverse_reshape(b_data2)


def main():
    text = "In computer science and operations research, a genetic algorithm (GA) is a metaheuristic inspired by the " \
           "process of natural selection that belongs to the larger class of evolutionary algorithms (EA)."

    # used for evaluate performance to generate random text of any length
    # text = ''.join(
    #    random.SystemRandom().choice(string.ascii_letters + string.digits + string.punctuation + string.whitespace) for
    #    _ in range(5000))

    print("Text:", text)

    # generate random key(it can have any length, could be the length of the plaintext)
    # in this case, I used 128 bit key
    key = str2bin(''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16)))

    print("Key:", len(key), key)

    # generate the encoding tables
    generate_pre_processing_tables()
    generate_mutation_tables()

    # get the ciphertext
    start = time()
    print("Final DNA sequence:", dna_get(text, key))
    end = time()

    print(end - start)


if __name__ == '__main__':
    main()
